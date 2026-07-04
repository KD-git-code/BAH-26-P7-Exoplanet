"""builder_tess_bls.py — Build a curated TESS dataset from raw MAST light curves.

Works entirely without any external catalogue (TOI, ExoFOP, etc.).
Labels are assigned automatically from BLS signal quality:
    0 = high-confidence periodic signal   (SNR ≥ SNR_CONFIRMED)
    1 = moderate signal / candidate       (SNR_CANDIDATE ≤ SNR < SNR_CONFIRMED)
    2 = no convincing periodic signal     (SNR < SNR_CANDIDATE)

Input  : raw_data/science_sector<N>.parquet
         Columns: target_id, sector, mission, label(-1), time[], flux[], flux_err[]

Outputs (in OUT_DIR):
    final_tess_dataset.pkl   — DataFrame with all scalar features (no view arrays)
    global_views.npy         — (N, 2000) float32
    local_views.npy          — (N, 201)  float32
    labels.npy               — (N,)      int64  (0/1/2 BLS-derived)
    scalar_features.npy      — (N, n_scalar_cols) float32
    missions.npy             — (N,)      object / str

All preprocessing from the original builder is preserved:
    ✓ Per-star sigma-clipping (MAD-based, 4σ)
    ✓ Savitzky-Golay detrending (24h window)
    ✓ Median-MAD normalisation
    ✓ Full BLS period search (log-spaced, fine duration grid)
    ✓ Phase folding centred on transit (BTJD-native, no offset needed)
    ✓ Median binning with NaN interpolation
    ✓ Robust out-of-transit normalisation for both views
    ✓ All 14 engineered transit-shape features
    ✓ Augmentation (phase jitter + flux noise, training mode only)
"""

import os
import pickle
import warnings
import argparse

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import median_abs_deviation, skew, kurtosis
from astropy.timeseries import BoxLeastSquares
import astropy.units as u

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════
# ── CONFIGURATION  (edit these, nothing else needs changing) ───────────────
# ═══════════════════════════════════════════════════════════════════════════

SECTOR       = 1
RAW_PATH     = f"raw_data/science_sector{SECTOR}.parquet"
OUT_DIR      = f"final_sector{SECTOR}_processed"

# ── View sizes ─────────────────────────────────────────────────────────────
GLOBAL_BINS        = 2000
LOCAL_BINS         = 201          # odd → one bin exactly at phase 0
LOCAL_DUR_MULTIPLE = 3.0          # local view = ±3 × transit duration

# ── BLS search grid ────────────────────────────────────────────────────────
PERIOD_MIN    = 0.5               # days
PERIOD_MAX    = 13.0              # days  (half of ~27d sector → ≥2 transits)
N_PERIODS     = 5000              # log-spaced trial periods
# Fine log-spaced duration grid (days). Upper bound < PERIOD_MIN avoids
# the duration > period crash that existed in earlier versions.
DURATION_GRID = np.geomspace(0.02, PERIOD_MIN * 0.85, 20)

# ── Preprocessing ──────────────────────────────────────────────────────────
SIGMA_CLIP  = 4.0                 # MAD-based clipping threshold
SG_WINDOW   = 721                 # Savitzky-Golay window (cadences). 721 ≈ 24h at 2-min
SG_POLY     = 3                   # SG polynomial order

# ── BLS-derived labelling thresholds ──────────────────────────────────────
# Tune these based on your validation results.
# These values are conservative — err on the side of more candidates.
SNR_CONFIRMED = 15.0              # label 0: high-SNR, almost certainly periodic
SNR_CANDIDATE =  7.0              # label 1: moderate SNR, worth investigating
                                  # label 2: SNR < SNR_CANDIDATE → no signal

# ── Augmentation ───────────────────────────────────────────────────────────
# Set BUILD_MODE = "science" to skip augmentation and keep label = -1
# (pure inference output). "train" applies augmentation and BLS labels.
BUILD_MODE       = "train"
N_AUGMENTS       = 2
PHASE_JITTER_STD = 0.01           # phase units (fraction of period)
FLUX_NOISE_STD   = 0.002          # normalised flux units

# ── Scalar feature columns saved to scalar_features.npy ───────────────────
# Only BLS-derived and engineered columns — no catalogue columns.
SCALAR_COLS = [
    "bls_period", "bls_depth", "bls_duration", "bls_snr",
    "transit_depth", "depth_snr", "ingress_egress_asymmetry",
    "flat_bottom_ratio", "oot_scatter", "transit_skewness",
    "secondary_depth", "secondary_primary_ratio",
    "odd_even_depth_diff", "odd_even_depth_ratio",
    "baseline_kurtosis", "n_transits", "duty_cycle",
    "duration_over_period",
    # flux_err statistics (extracted from raw flux_err array)
    "median_flux_err", "mean_flux_err",
    # Light-curve quality
    "n_cadences", "lc_span_days", "duty_cycle_obs",
]


# ═══════════════════════════════════════════════════════════════════════════
# ── PREPROCESSING UTILITIES ────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def sigma_clip(flux: np.ndarray, sigma: float = SIGMA_CLIP) -> np.ndarray:
    """Boolean mask — True where cadence is within sigma MADs of median."""
    med = np.nanmedian(flux)
    mad = median_abs_deviation(flux, nan_policy="omit")
    if mad == 0:
        return np.ones(len(flux), dtype=bool)
    return np.abs(flux - med) < sigma * 1.4826 * mad


def detrend(flux: np.ndarray) -> np.ndarray:
    """Savitzky-Golay detrend: divide flux by smoothed baseline.

    Window is capped to array length and kept odd. Returns flux unchanged
    if the array is too short for the polynomial order.
    """
    n      = len(flux)
    window = min(SG_WINDOW, n if n % 2 == 1 else n - 1)
    if window <= SG_POLY:
        return flux
    try:
        baseline = savgol_filter(flux, window_length=window, polyorder=SG_POLY)
        # Guard against near-zero baseline (would blow up division)
        baseline = np.where(np.abs(baseline) < 1e-9, 1.0, baseline)
        return flux / baseline
    except Exception:
        return flux


def median_bin(phase: np.ndarray, flux: np.ndarray,
               n_bins: int, phase_min: float, phase_max: float) -> np.ndarray:
    """Bin (phase, flux) into n_bins equal-width bins using the median.

    Empty bins are filled by linear interpolation over adjacent bins.
    """
    edges  = np.linspace(phase_min, phase_max, n_bins + 1)
    result = np.full(n_bins, np.nan, dtype=np.float32)
    for i in range(n_bins):
        mask = (phase >= edges[i]) & (phase < edges[i + 1])
        if mask.any():
            result[i] = np.nanmedian(flux[mask])

    nans = np.isnan(result)
    if nans.any() and not nans.all():
        idx        = np.arange(n_bins)
        result[nans] = np.interp(idx[nans], idx[~nans], result[~nans])
    elif nans.all():
        result[:] = 0.0
    return result


def robust_normalise(view: np.ndarray, transit_mask: np.ndarray) -> np.ndarray:
    """Normalise a binned view using only out-of-transit bins as baseline.

    Subtracts baseline median, divides by baseline MAD.
    transit_mask : bool array, True = in-transit bins (excluded from baseline).
    """
    baseline = view[~transit_mask]
    if len(baseline) == 0 or np.all(np.isnan(baseline)):
        baseline = view
    med = np.nanmedian(baseline)
    mad = median_abs_deviation(baseline, nan_policy="omit") * 1.4826
    if mad < 1e-9:
        mad = float(np.nanstd(baseline)) or 1.0
    return ((view - med) / mad).astype(np.float32)


def pad_or_trim(arr: np.ndarray, length: int) -> np.ndarray:
    if len(arr) >= length:
        return arr[:length]
    return np.concatenate([arr, np.zeros(length - len(arr), dtype=arr.dtype)])


# ═══════════════════════════════════════════════════════════════════════════
# ── BLS ────────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def run_bls(time: np.ndarray, flux: np.ndarray):
    """Run BLS and return (period, t0, depth, duration, snr).

    All values are np.nan on failure. t0 is in the same time system as
    `time` (BTJD for TESS) — no external offset correction needed.
    """
    try:
        model   = BoxLeastSquares(time * u.day, flux)
        periods = np.geomspace(PERIOD_MIN, PERIOD_MAX, N_PERIODS)
        result  = model.power(
            periods        * u.day,
            duration=DURATION_GRID * u.day,
            method="fast",
            objective="snr",
        )
        best     = int(np.argmax(result.power))
        period   = float(result.period[best].value)
        t0       = float(result.transit_time[best].value)
        depth    = float(abs(result.depth[best]))
        duration = float(result.duration[best].value)
        snr      = float(result.power[best])
        return period, t0, depth, duration, snr
    except Exception as exc:
        print(f"    [BLS ERROR] {type(exc).__name__}: {exc}")
        return np.nan, np.nan, np.nan, np.nan, np.nan


def bls_label(snr: float) -> int:
    """Assign a training label from BLS SNR.

    0 → high-confidence periodic signal   (SNR ≥ SNR_CONFIRMED)
    1 → moderate signal / candidate       (SNR_CANDIDATE ≤ SNR < SNR_CONFIRMED)
    2 → no convincing periodic signal     (SNR < SNR_CANDIDATE or nan)
    """
    if not np.isfinite(snr):
        return 2
    if snr >= SNR_CONFIRMED:
        return 0
    if snr >= SNR_CANDIDATE:
        return 1
    return 2


# ═══════════════════════════════════════════════════════════════════════════
# ── ENGINEERED FEATURES ────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def extract_engineered_features(time: np.ndarray, flux: np.ndarray,
                                 period: float, t0: float,
                                 duration: float) -> dict:
    """Extract 14 physically motivated scalar features.

    All features are robust to NaN — missing values return np.nan rather
    than raising. The full feature set:

    Transit shape
        transit_depth, depth_snr, ingress_egress_asymmetry,
        flat_bottom_ratio, oot_scatter, transit_skewness

    Secondary eclipse
        secondary_depth, secondary_primary_ratio

    Odd-even asymmetry (eclipsing binary discriminant)
        odd_even_depth_diff, odd_even_depth_ratio

    Baseline statistics
        baseline_kurtosis

    Period-related
        n_transits, duty_cycle, duration_over_period
    """
    feats = {}

    # Phase relative to transit centre, ∈ (−0.5, 0.5]
    phase       = ((time - t0) / period + 0.5) % 1.0 - 0.5
    half_dur_ph = (duration / period) / 2.0       # transit half-width in phase

    in_transit  = np.abs(phase) < half_dur_ph
    out_transit = ~in_transit

    # ── Transit shape ──────────────────────────────────────────────────────
    if in_transit.sum() > 5 and out_transit.sum() > 10:
        t_flux = flux[in_transit]
        b_flux = flux[out_transit]
        b_med  = float(np.nanmedian(b_flux))
        b_std  = float(np.nanstd(b_flux))

        feats["transit_depth"] = float(b_med - np.nanmedian(t_flux))
        feats["depth_snr"]     = (feats["transit_depth"] / b_std
                                  if b_std > 0 else 0.0)

        # Ingress vs egress asymmetry
        left  = flux[(phase > -half_dur_ph) & (phase < 0)]
        right = flux[(phase > 0)            & (phase < half_dur_ph)]
        feats["ingress_egress_asymmetry"] = (
            float(np.nanmedian(left) - np.nanmedian(right))
            if len(left) > 2 and len(right) > 2 else 0.0
        )

        # Flat-bottom ratio: fraction of in-transit bins below 90% of depth
        flat_thresh = feats["transit_depth"] * 0.9
        flat_bins   = t_flux < (b_med - flat_thresh)
        feats["flat_bottom_ratio"] = float(flat_bins.sum() / max(len(t_flux), 1))

        feats["oot_scatter"]      = b_std
        feats["transit_skewness"] = float(skew(t_flux))

    else:
        for k in ["transit_depth", "depth_snr", "ingress_egress_asymmetry",
                  "flat_bottom_ratio", "oot_scatter", "transit_skewness"]:
            feats[k] = np.nan

    # ── Secondary eclipse (phase 0.5) ──────────────────────────────────────
    sec_mask = np.abs(phase - 0.5) < half_dur_ph
    if sec_mask.sum() > 3 and out_transit.sum() > 10:
        b_med = float(np.nanmedian(flux[out_transit]))
        feats["secondary_depth"] = float(b_med - np.nanmedian(flux[sec_mask]))
        p_depth = feats.get("transit_depth", np.nan)
        feats["secondary_primary_ratio"] = (
            float(feats["secondary_depth"] / p_depth)
            if np.isfinite(p_depth) and p_depth > 0 else np.nan
        )
    else:
        feats["secondary_depth"]         = np.nan
        feats["secondary_primary_ratio"] = np.nan

    # ── Odd-even transit depth asymmetry ───────────────────────────────────
    # Fold at half the period. If a secondary appears, the signal is likely
    # an eclipsing binary with P_true = 2 × BLS period.
    transit_num = np.floor((time - t0) / period).astype(int)
    phase_half  = ((time - t0) / (period / 2.0) + 0.5) % 1.0 - 0.5
    odd_mask    = (np.abs(phase_half) < half_dur_ph) & (transit_num % 2 == 0)
    even_mask   = (np.abs(phase_half) < half_dur_ph) & (transit_num % 2 == 1)

    if odd_mask.sum() > 3 and even_mask.sum() > 3 and out_transit.sum() > 10:
        b_med      = float(np.nanmedian(flux[out_transit]))
        odd_depth  = float(b_med - np.nanmedian(flux[odd_mask]))
        even_depth = float(b_med - np.nanmedian(flux[even_mask]))
        mean_depth = abs(odd_depth + even_depth) / 2.0
        feats["odd_even_depth_diff"]  = float(abs(odd_depth - even_depth))
        feats["odd_even_depth_ratio"] = float(
            abs(odd_depth - even_depth) / max(mean_depth, 1e-9)
        )
    else:
        feats["odd_even_depth_diff"]  = np.nan
        feats["odd_even_depth_ratio"] = np.nan

    # ── Baseline kurtosis (excess kurtosis of out-of-transit flux) ─────────
    feats["baseline_kurtosis"] = (
        float(kurtosis(flux[out_transit]))
        if out_transit.sum() > 10 else np.nan
    )

    # ── Period-related features ────────────────────────────────────────────
    time_span             = float(np.nanmax(time) - np.nanmin(time))
    feats["n_transits"]          = float(time_span / period) if period > 0 else np.nan
    feats["duty_cycle"]          = float(duration / period)  if period > 0 else np.nan
    feats["duration_over_period"] = feats["duty_cycle"]       # alias kept for compat

    return feats


# ═══════════════════════════════════════════════════════════════════════════
# ── PHASE FOLDING & VIEW CONSTRUCTION ──────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def make_views(time: np.ndarray, flux: np.ndarray,
               period: float, t0: float,
               duration: float):
    """Build global (GLOBAL_BINS,) and local (LOCAL_BINS,) phase-folded views.

    Global view : entire phase range (−0.5, 0.5], uniformly binned.
    Local view  : ±LOCAL_DUR_MULTIPLE × transit duration, zoomed in.

    Both views are robust-normalised using out-of-transit bins as baseline.
    t0 must be in the same time system as `time` — no external offset needed
    because BLS returns t0 directly in BTJD.
    """
    phase = ((time - t0) / period + 0.5) % 1.0 - 0.5
    order = np.argsort(phase)
    phase = phase[order]
    flux  = flux[order]

    # ── Global view ───────────────────────────────────────────────────────
    global_raw  = median_bin(phase, flux, GLOBAL_BINS, -0.5, 0.5)
    bin_centres = (np.arange(GLOBAL_BINS) + 0.5) / GLOBAL_BINS - 0.5
    half_dur_ph = max(duration, 0.02) / period
    transit_mask = np.abs(bin_centres) < half_dur_ph
    global_view  = robust_normalise(global_raw, transit_mask)

    # ── Local view ────────────────────────────────────────────────────────
    half_window = LOCAL_DUR_MULTIPLE * half_dur_ph
    half_window = np.clip(half_window, 0.02, 0.48)  # never wider than half-orbit

    local_mask  = np.abs(phase) <= half_window
    l_phase     = phase[local_mask]
    l_flux      = flux[local_mask]

    if l_phase.size < 10:
        local_view = np.zeros(LOCAL_BINS, dtype=np.float32)
    else:
        local_raw = median_bin(l_phase, l_flux, LOCAL_BINS,
                               -half_window, half_window)
        # Use edge bins (outer 10% on each side) as out-of-transit baseline
        n_edge        = max(1, LOCAL_BINS // 10)
        edge_mask     = np.zeros(LOCAL_BINS, dtype=bool)
        edge_mask[:n_edge]  = True
        edge_mask[-n_edge:] = True
        # edge_mask marks the OOT region; robust_normalise excludes transit_mask
        # transit_mask argument = ~edge_mask (True = in-transit / exclude)
        local_view = robust_normalise(local_raw, ~edge_mask)

    return (pad_or_trim(global_view, GLOBAL_BINS),
            pad_or_trim(local_view,  LOCAL_BINS))


# ═══════════════════════════════════════════════════════════════════════════
# ── AUGMENTATION ───────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def augment_views(global_view: np.ndarray, local_view: np.ndarray,
                  rng: np.random.Generator) -> list[tuple]:
    """Generate N_AUGMENTS (global, local) pairs per real sample.

    Augmentations:
      • Phase jitter: circular shift of global view by a small random
        integer number of bins. Local view left un-shifted (already centred).
      • Flux noise: Gaussian noise scaled to FLUX_NOISE_STD added to both.
    """
    augmented = []
    for _ in range(N_AUGMENTS):
        jitter  = int(rng.normal(0, PHASE_JITTER_STD * GLOBAL_BINS))
        g_aug   = np.roll(global_view, jitter).copy().astype(np.float32)
        g_aug  += rng.normal(0, FLUX_NOISE_STD, size=g_aug.shape).astype(np.float32)
        l_aug   = (local_view
                   + rng.normal(0, FLUX_NOISE_STD,
                                size=local_view.shape).astype(np.float32))
        augmented.append((g_aug, l_aug.astype(np.float32)))
    return augmented


# ═══════════════════════════════════════════════════════════════════════════
# ── PER-STAR PROCESSOR ─────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def process_one(row: dict, rng: np.random.Generator) -> list[dict] | None:
    """Process one raw parquet row → list of dataset entries (real + augments).

    Steps
    ─────
    1. Validate & cast raw arrays
    2. Sigma-clip outlier cadences
    3. Savitzky-Golay detrend
    4. Median-MAD normalise (whole light curve)
    5. BLS period search
    6. Assign BLS-derived label (train mode) or keep -1 (science mode)
    7. Extract engineered features
    8. Build global & local phase-folded views
    9. Augment (train mode only)

    Returns None if any critical step fails.
    """
    tic_id = row.get("target_id")
    try:
        time     = np.asarray(row["time"],     dtype=np.float64)
        flux     = np.asarray(row["flux"],     dtype=np.float64)
        flux_err = np.asarray(row["flux_err"], dtype=np.float64)

        # Quick quality gate
        if len(time) < 200:
            return None

        # ── 1. Raw flux-err statistics (before any clipping) ───────────────
        median_ferr = float(np.nanmedian(flux_err))
        mean_ferr   = float(np.nanmean(flux_err))
        n_raw       = len(time)
        lc_span     = float(time[-1] - time[0])

        # Fraction of time with actual observations (duty cycle of the LC)
        # 2-min cadence → one point every 2/1440 days
        expected_n  = lc_span / (2.0 / 1440.0)
        duty_obs    = n_raw / expected_n if expected_n > 0 else np.nan

        # ── 2. Sigma-clip ──────────────────────────────────────────────────
        good = sigma_clip(flux)
        time, flux, flux_err = time[good], flux[good], flux_err[good]
        if len(time) < 200:
            return None

        # ── 3. Detrend ─────────────────────────────────────────────────────
        flux = detrend(flux)

        # ── 4. Normalise ───────────────────────────────────────────────────
        med = float(np.nanmedian(flux))
        if abs(med) < 1e-9:
            return None
        flux = flux / med

        # ── 5. BLS ─────────────────────────────────────────────────────────
        bls_period, t0, bls_depth, bls_dur, bls_snr = run_bls(time, flux)

        if not np.isfinite(bls_period) or not np.isfinite(t0):
            return None         # BLS failed entirely — skip this star

        period   = bls_period
        duration = bls_dur if np.isfinite(bls_dur) else 0.1

        # ── 6. Label assignment ─────────────────────────────────────────────
        if BUILD_MODE == "train":
            label = bls_label(bls_snr)
        else:
            label = -1          # science/inference mode — no label

        # ── 7. Engineered features ─────────────────────────────────────────
        eng = extract_engineered_features(time, flux, period, t0, duration)

        # ── 8. Phase-folded views ──────────────────────────────────────────
        global_view, local_view = make_views(time, flux, period, t0, duration)

        base = {
            # Identity
            "target_id":    int(tic_id),
            "sector":       int(row.get("sector", -1)),
            "mission":      str(row.get("mission", "TESS")),
            "label":        label,
            "augmented":    False,
            # Fold parameters
            "fold_period":   float(period),
            "fold_t0":       float(t0),
            "fold_duration": float(duration),
            # BLS features
            "bls_period":   float(bls_period),
            "bls_depth":    float(bls_depth)  if np.isfinite(bls_depth)  else np.nan,
            "bls_duration": float(bls_dur)    if np.isfinite(bls_dur)    else np.nan,
            "bls_snr":      float(bls_snr)    if np.isfinite(bls_snr)    else np.nan,
            # Flux-error / LC-quality features
            "median_flux_err": median_ferr,
            "mean_flux_err":   mean_ferr,
            "n_cadences":      float(n_raw),
            "lc_span_days":    lc_span,
            "duty_cycle_obs":  float(duty_obs),
            # Views (dropped before DataFrame save; saved as .npy)
            "global_view": global_view,
            "local_view":  local_view,
            # Engineered features
            **{k: (float(v) if np.isfinite(float(v)) else np.nan)
               for k, v in eng.items()},
        }

        # ── 9. Augmentation ─────────────────────────────────────────────────
        results = [base]
        if BUILD_MODE == "train":
            for g_aug, l_aug in augment_views(global_view, local_view, rng):
                aug = base.copy()
                aug["augmented"]   = True
                aug["global_view"] = g_aug
                aug["local_view"]  = l_aug
                results.append(aug)

        return results

    except Exception as exc:
        print(f"  [ERROR] TIC {tic_id} — {type(exc).__name__}: {exc}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# ── MAIN BUILD FUNCTION ────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def build(raw_path: str = RAW_PATH, out_dir: str = OUT_DIR,
          sector: int = SECTOR) -> tuple:
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(42)

    print(f"\n{'='*60}")
    print(f"  TESS BLS Dataset Builder")
    print(f"  Sector  : {sector}")
    print(f"  Input   : {raw_path}")
    print(f"  Output  : {out_dir}/")
    print(f"  Mode    : {BUILD_MODE}")
    print(f"  SNR thresholds: confirmed≥{SNR_CONFIRMED}, candidate≥{SNR_CANDIDATE}")
    print(f"{'='*60}\n")

    print(f"Loading {raw_path}...")
    df_raw = pd.read_parquet(raw_path)
    print(f"  {len(df_raw)} raw light curves\n")

    all_records: list[dict] = []
    skipped = 0

    for i, row in enumerate(df_raw.to_dict("records")):
        results = process_one(row, rng)
        if results is None:
            skipped += 1
        else:
            all_records.extend(results)

        if (i + 1) % 50 == 0 or (i + 1) == len(df_raw):
            real_so_far = sum(1 for r in all_records if not r["augmented"])
            print(f"  [{i+1:4d}/{len(df_raw)}]  "
                  f"{real_so_far} real | "
                  f"{len(all_records)-real_so_far} augmented | "
                  f"{skipped} skipped")

    print(f"\nProcessing complete.")
    print(f"  Real records    : {sum(1 for r in all_records if not r['augmented'])}")
    print(f"  Augmented       : {sum(1 for r in all_records if r['augmented'])}")
    print(f"  Total entries   : {len(all_records)}")
    print(f"  Skipped         : {skipped}")

    if not all_records:
        print("\n[FATAL] No records produced — check your raw parquet file.")
        return None, None, None, None

    # ── Assemble DataFrame ─────────────────────────────────────────────────
    out_df = pd.DataFrame(all_records)

    # Stack view arrays into numpy matrices
    global_views = np.stack(out_df["global_view"].values).astype(np.float32)
    local_views  = np.stack(out_df["local_view"].values).astype(np.float32)
    labels       = out_df["label"].values.astype(np.int64)
    missions     = out_df["mission"].values

    # Scalar features — use only columns that are actually populated
    available_scalar = [c for c in SCALAR_COLS if c in out_df.columns
                        and out_df[c].notna().any()]
    missing_scalar   = [c for c in SCALAR_COLS if c not in available_scalar]
    if missing_scalar:
        print(f"\n  [INFO] Scalar cols with no data (skipped): {missing_scalar}")
    scalar_df    = out_df[available_scalar].apply(pd.to_numeric, errors="coerce")
    scalar_feats = scalar_df.values.astype(np.float32)

    # Drop view arrays from DataFrame (saved separately as .npy)
    df_save = out_df.drop(columns=["global_view", "local_view"])

    # ── Save ───────────────────────────────────────────────────────────────
    print(f"\nSaving outputs to {out_dir}/")

    pkl_path = os.path.join(out_dir, "final_tess_dataset.pkl")
    with open(pkl_path, "wb") as fh:
        pickle.dump(df_save, fh)
    print(f"  {pkl_path}")

    for name, arr in [
        ("global_views",    global_views),
        ("local_views",     local_views),
        ("labels",          labels),
        ("missions",        missions),
        ("scalar_features", scalar_feats),
    ]:
        path = os.path.join(out_dir, f"{name}.npy")
        np.save(path, arr)
        print(f"  {path}  shape={arr.shape}")

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Dataset summary")
    print(f"  Global views : {global_views.shape}")
    print(f"  Local views  : {local_views.shape}")
    print(f"  Scalar feats : {scalar_feats.shape}  ({len(available_scalar)} cols)")

    if BUILD_MODE == "train":
        print(f"\n  Label distribution (BLS-derived, train + augmented):")
        names = {0: "Confirmed (high SNR)", 1: "Candidate (mod. SNR)", 2: "No signal"}
        for lbl, name in names.items():
            real_n = int(((labels == lbl) & ~out_df["augmented"].values).sum())
            aug_n  = int(((labels == lbl) &  out_df["augmented"].values).sum())
            print(f"    {lbl} {name:<28}: {real_n} real + {aug_n} augmented "
                  f"= {real_n+aug_n} total")

        bls_snrs = df_save.loc[~df_save["augmented"], "bls_snr"]
        print(f"\n  BLS SNR stats (real records only):")
        print(f"    min={bls_snrs.min():.2f}  median={bls_snrs.median():.2f}"
              f"  max={bls_snrs.max():.2f}")
    else:
        print(f"\n  Science/inference mode — all labels = -1")

    print(f"{'='*60}\n")
    print(f"DataFrame info:")
    print(df_save.info())

    return df_save, global_views, local_views, labels


# ═══════════════════════════════════════════════════════════════════════════
# ── CLI ────────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Build curated TESS dataset from raw MAST light curves "
                    "(no external catalogue required)."
    )
    p.add_argument("--raw",    default=RAW_PATH,
                   help=f"Path to raw parquet (default: {RAW_PATH})")
    p.add_argument("--out",    default=OUT_DIR,
                   help=f"Output directory (default: {OUT_DIR})")
    p.add_argument("--sector", type=int, default=SECTOR,
                   help=f"Sector number for labelling output paths (default: {SECTOR})")
    p.add_argument("--mode",   choices=["train", "science"], default=BUILD_MODE,
                   help="train=BLS labels+augmentation, science=label -1, no augmentation")
    p.add_argument("--snr-confirmed", type=float, default=SNR_CONFIRMED,
                   help=f"BLS SNR threshold for label 0 (default: {SNR_CONFIRMED})")
    p.add_argument("--snr-candidate", type=float, default=SNR_CANDIDATE,
                   help=f"BLS SNR threshold for label 1 (default: {SNR_CANDIDATE})")
    args = p.parse_args()

    # Apply CLI overrides
    BUILD_MODE      = args.mode
    SNR_CONFIRMED   = args.snr_confirmed
    SNR_CANDIDATE   = args.snr_candidate

    build(raw_path=args.raw, out_dir=args.out, sector=args.sector)