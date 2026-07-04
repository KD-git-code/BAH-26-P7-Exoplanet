# TESS Exoplanet Detection Pipeline — Technical Documentation

## Overview

This document describes the complete data collection and preprocessing pipeline developed for the TESS exoplanet detection project. The pipeline transforms raw TESS light curve data from the MAST archive into structured, model-ready datasets suitable for both unsupervised pretraining and supervised classification.

The pipeline operates in two distinct modes. Science mode processes unlabeled light curves from a full TESS sector for inference and autoencoder pretraining. Train mode processes labeled light curves from a competition-provided curated dataset for supervised model training. This document covers the science mode pipeline in full, as that is what has been implemented and validated.

### Architecture Overview

MAST Archive (STScI)
│
▼
collect_science.py ← Downloads raw light curves, one per star
│
│ raw_data/science_sector{N}.parquet
│ (one row per star, time/flux arrays stored as objects)
▼
builder_tess.py ← Processes every star through the full chain
│
├── sigma_clip()
├── detrend()
├── normalize()
├── run_bls()
├── extract_engineered_features()
├── make_views() [global + local phase-fold]
└── augment_views()
│
│ final_sector1_processed/
▼

### File Shapes

| File name             | Size                         |
| --------------------- | ---------------------------- |
| _global_views.npy_    | (N, 2000)                    |
| _local_views.npy_     | (N, 201)                     |
| _scalar_features.npy_ | (N, 23)                      |
| _labels.npy_          | (N,) all -1 for science mode |
| _missions.npy_        | (N,)                         |

_final_tess_dataset.pkl_ == full dataframe with all metadata

## Part 1 — Data Collection (collect_science.py)

### 1.1 Source

All light curves are downloaded from the MAST archive (Mikulski Archive for Space Telescopes) hosted by the Space Telescope Science Institute at archive.stsci.edu. Specifically, we use the TESS 2-minute cadence light curve products — pre-reduced .fits files produced by the official TESS Science Processing Operations Center (SPOC) pipeline.

We do not use the TOI (TESS Objects of Interest) catalog or any pre-labeled disposition data. The science dataset is entirely unlabeled — it represents a blind survey of stars observed by TESS in a given sector with no prior knowledge of which stars host planets.

### 1.2 Target Selection

TESS observes the sky in sectors of approximately 27 days each. Within each sector, roughly 20,000 stars are monitored at 2-minute cadence (the remainder are captured only in 10-minute Full Frame Images). We download only the 2-minute cadence targets because:

720 cadences per day provides sufficient time resolution to detect transits as short as 1 hour.

The SPOC pipeline produces PDCSAP (Pre-search Data Conditioning Simple Aperture Photometry) flux for these targets, with instrumental systematics already removed
BLS period searches require dense time sampling to reliably detect short-period signals

Target IDs for a given sector are retrieved by querying the MAST Observations API:

```python
obs = Observations.query_criteria(
    obs_collection = "TESS",
    dataproduct_type = "timeseries",
    sequence_number = sector,
    t_exptime = [100, 140], # 120s ± buffer for 2-min cadence
)
```

This returns TIC (TESS Input Catalog) IDs for all 2-minute targets in the requested sector.

### 1.3 Light Curve Download

Each star is downloaded individually using lightkurve, which handles MAST authentication, .fits file retrieval, and basic data extraction:

```python
results = lk.search_lightcurve(
    f"TIC {tic_id}", mission="TESS",
    exptime=120, sector=sector
)
lc = results[0].download()
```

The downloaded LightCurve object contains:

time — timestamps in BTJD (Barycentric TESS Julian Date = BJD − 2,457,000)
flux — PDCSAP flux in electrons/second
flux_err — per-cadence uncertainty
quality — bitmask encoding flagged cadences

A minimum quality check discards any star with fewer than 100 clean cadences after NaN and outlier removal.

### 1.4 What Gets Stored in the Parquet

After download, each star's data is serialized into a single row of a Parquet file. This is the key design decision that compresses what would be 18,000+ individual rows of raw cadence data into a single structured record per star.

The raw light curve arrays (time, flux) are stored as Python objects (numpy arrays) inside Parquet's object columns. This is non-standard but necessary — a typical relational representation would require a separate table for cadence data, which would make downstream processing far more complex.

Each row contains

| Column    | Type   | Description                                    |
| --------- | ------ | ---------------------------------------------- |
| target_id | int64  | TIC Identifier                                 |
| sector    | int64  | TESS sector number                             |
| mission   | str    | Always "TESS" for this pipeline                |
| label     | int64  | -1 (unlabeled)                                 |
| time      | object | numpy float64 array, BTJD timestamps           |
| flux      | object | numpy float32 array, PDCSAP normalized flux    |
| flux_err  | object | numpy float32 array, per-cadence uncertainties |

### 1.5 Parallel Download Architecture

Downloading 1,732 stars sequentially would take several hours. The collector uses `ThreadPoolExecutor` with `MAX_WORKERS = 4` parallel threads. Threading (not multiprocessing) is used because the bottleneck is network I/O, waiting for MAST to respond, not CPU computation. Python's GIL does not block concurrent I/O operations.

A thread lock (`threading.Lock`) wraps all print statements to prevent Windows stdout race conditions, which previously caused ValueError: I/O operation on closed file crashes when multiple threads printed simultaneously.

Checkpointing saves completed records every 25 stars, making the collector resumable after crashes:

```python
if records and len(records) % CHECKPOINT_EVERY == 0:
    with open(ckpt, "wb") as fh:
        pickle.dump(records, fh)
```

### 1.6 Why Parquet

The choice of Parquet over CSV or pickle for the raw data stage has several practical motivations:

- **Column pruning**. The builder reads only the columns it needs without loading the full file into memory. For a dataset with 1,732 stars and large array columns this matters.

- **Type preservatio**n. Parquet stores dtypes explicitly. Unlike CSV, float64 arrays are read back as float64, not strings requiring re-parsing.

- **Compression**. Parquet with snappy compression reduces file size by roughly 3-4x compared to uncompressed pickle for this data structure.

- **Partial reads.** The builder can read metadata columns (target_id, label, catalog_period) without deserializing the full time/flux arrays, enabling fast catalog operations before the heavy processing begins.

## Part 2 — The Build Pipeline (`main_builder.py`)

The builder reads the raw Parquet file and processes every star through a sequential chain of six stages. The output is a set of numpy arrays and a metadata dataframe ready for model training or inference.

### 2.1 Configuration

All tunable parameters are centralized in config.py:

```python
PERIOD_MIN = 0.5 # days — BLS search lower bound
PERIOD_MAX = 15.0 # days — BLS search upper bound
GLOBAL_BINS = 2000 # phase bins for global view
LOCAL_BINS = 200 # phase bins for local view
LOCAL_FRAC = 0.1 # ±10% of phase around transit centre for local view
N_AUGMENTS = 2 # augmented copies per real sample
SG_WINDOW = 901 # Savitzky-Golay detrend window (cadences)
SG_POLY = 2 # Savitzky-Golay polynomial order
SIGMA_UPPER = 5.0 # sigma clip threshold (upper)
SIGMA_LOWER = 5.0 # sigma clip threshold (lower)
BTJD_OFFSET = 2457000.0 # BJD → BTJD conversion
```

### 2.2 Stage 1 — Sigma Clipping (`sigma_clip`)

Raw TESS photometry contains outlier cadences from cosmic rays, stellar flares, and detector artifacts that survived the quality flag filtering. These are removed using iterative sigma clipping:

```python
def sigma_clip(flux, sigma_upper=SIGMA_UPPER, sigma_lower=SIGMA_LOWER, n_iter=3):
    mask = np.ones(len(flux), dtype=bool)
    for * in range(n*iter):
    med = np.nanmedian(flux[mask])
    std = np.nanstd(flux[mask])
    mask &= (flux >= med - sigma_lower * std) & \
        (flux <= med + sigma*upper * std)
    return mask
```

Three iterations are used. On the first pass the median and std are computed over all cadences. Cadences beyond 5σ are masked. The second and third passes recompute statistics on the surviving cadences, progressively tightening the clip around the true baseline. This iterative approach prevents a single extreme outlier from inflating the std and allowing other outliers to survive.

The asymmetric threshold (same value for upper and lower here, but configurable separately) matters because stellar flares produce sharp upward spikes while instrumental artifacts can go either direction. A tighter upper threshold would be appropriate if flare removal were a specific concern.

### 2.3 Stage 2 — Detrending (detrend)

Even after PDCSAP systematics correction, some light curves retain slow trends from residual instrumental effects, stellar rotation, or imperfect background subtraction. These are removed using a **_Savitzky-Golay_** filter:

```python
def detrend(time, flux):
    window = min(SG*WINDOW, len(flux) if len(flux) % 2 == 1 else len(flux) - 1)
    if window <= SG_POLY:
        return flux
    baseline = savgol_filter(flux, window_length=window, polyorder=SG_POLY)
    baseline = np.where(np.abs(baseline) < 1e-6, 1.0, baseline)
    return flux / baseline
```

The Savitzky-Golay filter fits a polynomial of degree `SG_POLY = 2` to a sliding window of `SG_WINDOW = 901` cadences (~30 hours at 2-minute cadence). This window is wide enough to smooth over stellar variability on timescales shorter than typical transit durations while preserving transit signals themselves, which appear as sharp local dips much narrower than the window.

Dividing by the baseline (rather than subtracting) normalizes the flux to fractional units, making the representation consistent across stars of very different brightnesses. The 1e-6 guard prevents division by zero at poorly-estimated baseline points.

### 2.4 Stage 3 — Normalization (normalize)

After detrending, the flux is normalized to zero median and unit MAD (Median Absolute Deviation):

```python
def normalize_flux(flux):
    median = np.nanmedian(flux)
    mad = np.nanmedian(np.abs(flux - median))
    if mad == 0 or np.isnan(mad):
    return flux - median
    return (flux - median) / mad
```

MAD normalization is preferred over standard deviation normalization because MAD is robust to outliers. A single deep eclipse event would dramatically inflate the standard deviation, causing the entire light curve to be compressed to near-zero values. MAD uses the median of absolute deviations, which is insensitive to the few cadences inside a transit.

The result is a flux array centered at approximately zero with most out-of-transit values in the range [-3, 3], and transit dips appearing as negative excursions below zero.

### 2.5 Stage 4 — BLS Period Search (run_bls)

The Box Least Squares algorithm (Kovács et al. 2002) searches for the best-fit periodic box-shaped signal in the detrended, normalized flux. It is implemented using `astropy.timeseries.BoxLeastSquares`:

```python
def run_bls(time, flux):
model = BoxLeastSquares(time * u.day, flux)
periods = np.linspace(PERIOD*MIN, PERIOD_MAX, 2000)
result = model.power(
periods \* u.day,
duration = [0.05, 0.1, 0.15, 0.2] \* u.day,
method = 'fast'
)
best = np.argmax(result.power)
return (
    float(result.period[best].value),
    float(result.transit_time[best].value),
    float(result.depth[best]),
    float(result.duration[best].value),
    float(result.power[best]),
    )
```

The search sweeps 2,000 evenly-spaced periods between 0.5 and 13 days. Four transit durations are tested at each period (0.05, 0.1, 0.15, 0.2 days = 1.2, 2.4, 3.6, 4.8 hours). The fast method uses an optimized C implementation.

A critical bug was identified and fixed during development. The original implementation used model.autopower() which internally updates the best_period variable but the code was reading from a stale reference that was not updated after the power computation. This caused all stars to return the same period regardless of their actual BLS result. The fix explicitly indexes into the result arrays using np.argmax(result.power), which directly identifies the best period from the computed power spectrum on each call.

The BLS outputs for each star are:

- bls_period — best-fit orbital period in days
- bls_t0 — time of first transit in BTJD
- bls_depth — fractional flux decrease during transit
- bls_duration — transit duration in days
- bls_snr — signal power at the best period (proxy for detection significance)

### 2.6 Stage 4b — Engineered Feature Extraction (extract_engineered_features)

Beyond the four BLS scalar outputs, a set of physically motivated diagnostic features are computed from the raw light curve folded at the BLS period. These features directly implement the false-positive discrimination tests used in published vetting pipelines such as ExoMiner.
The function receives (time, flux, period, t0, duration) and computes the following:

#### Transit Shape Features

The light curve is phase-folded at the BLS period and split into in-transit and out-of-transit windows:

```python
phase = ((time - t0) / period + 0.5) % 1.0 - 0.5
in_transit = np.abs(phase) < (duration / period / 2)
out_transit = ~in_transit
```

- `transit_depth` — Median baseline minus median in-transit flux. A robust depth estimate that resists outliers better than the BLS depth, which assumes a perfect box.

- `depth_snr` — Transit depth divided by out-of-transit standard deviation scaled by the square root of the number of in-transit points. The standard per-transit SNR metric used across the field.

- `ingress_egress_asymmetry` — Difference between median flux in the left half of the transit window (ingress) and the right half (egress). A symmetric planetary transit produces a value near zero. Significant asymmetry indicates either a V-shaped grazing eclipse, stellar variability coinciding with the transit window, or a poorly-estimated t0.

- `flat_bottom_ratio` — Fraction of in-transit points deeper than 90% of the transit depth. A planetary transit with a flat bottom (planet fully inside stellar disk) has a high flat-bottom ratio. A V-shaped grazing eclipsing binary has a near-zero flat-bottom ratio. This is one of the most powerful single features for distinguishing planet transits from grazing eclipses.

- `oot_scatter` — Standard deviation of the out-of-transit flux. Measures the photometric noise floor for this specific star. Directly determines the minimum detectable transit depth.

- `transit_skewness` — Statistical skewness of the flux distribution inside the transit window. A symmetric transit produces near-zero skewness. Asymmetric events (flares, stellar spots crossing, blended variability) produce non-zero skewness.

#### Secondary Eclipse Features

The secondary eclipse check tests for a flux dip at phase 0.5 — the point in the orbit diametrically opposite the primary transit. A planet produces no secondary eclipse (it disappears behind the star, but planets do not emit significant light at TESS wavelengths). An eclipsing binary produces a secondary eclipse when the secondary star passes behind the primary:

```python
secondary_mask = np.abs(phase - 0.5) < (duration / period / 2)
secondary_depth = median(baseline) - median(flux[secondary_mask])
```

- `secondary_depth` — Flux dip at phase 0.5. Near zero for planets, positive for eclipsing binaries.
- `secondary_primary_ratio` — Secondary depth divided by primary depth. Values above 0.1 are flagged as likely eclipsing binary signatures. Values above 0.5 are strongly indicative. An equal-depth secondary (ratio ≈ 1.0) indicates two stars of similar temperature — a definitive eclipsing binary.

#### Odd-Even Depth Asymmetry

For an eclipsing binary observed at half its true period (a common BLS failure mode), alternating transits correspond to alternating primary and secondary eclipses, which have different depths. For a genuine planet, all transits are identical:

```python
odd_mask = in_transit_mask & (transit_number % 2 == 0)
even_mask = in_transit_mask & (transit_number % 2 == 1)
odd_depth = baseline - median(flux[odd_mask])
even_depth = baseline - median(flux[even_mask])
```

- `odd_even_depth_diff` — Absolute difference between odd and even transit depths. Near zero for planets, positive for period-aliased EBs.
- `odd_even_depth_ratio` — Odd-even depth difference normalized by mean depth. Values above 0.1 are flagged. Note: this feature requires clipping at 10 before model input because near-zero denominators (very shallow transits) produce extreme ratio values. Maximum observed in Sector 1 was 448.5, caused by a star with near-zero mean transit depth where numerical noise dominates.

#### Baseline Quality Features

- `baseline_kurtosis` — Excess kurtosis of the out-of-transit flux distribution. Near-Gaussian baseline noise produces kurtosis near zero. High kurtosis indicates heavy tails — either remaining outliers after sigma clipping or a genuinely variable star whose variability contaminates the "baseline." Also requires clipping at 20 before model input (maximum observed: 365).

#### Period and Geometry Features

- `n_transits` — Number of transit events observed, computed as light curve span divided by period. More transits means higher statistical confidence in the detection and more robust depth/shape estimates.

- `duty_cycle / duration_over_period` — Transit duration as a fraction of the orbital period. Physically constrained: planetary transits have duty cycles of 0.1-5% for most orbital configurations. Very high duty cycles (> 10%) indicate either very long durations relative to period (geometrically implausible for most planet/star size ratios) or a misidentified period.

- Note: `duty_cycle` and `duration_over_period` are identical values stored under two column names for legacy compatibility. One will be dropped before model input.

#### Light Curve Metadata Features

Three additional features characterize the quality of the observation rather than the transit signal:

- `median_flux_err` / mean_flux_err — Per-cadence photometric uncertainty from the SPOC pipeline. Reflects the photon noise floor plus read noise. Higher values indicate fainter stars or noisier detector regions.
- `n_cadences` — Total number of cadences after quality filtering. Typically 17,000-18,279 for a complete Sector 1 observation.
- `lc_span_days` — Total observation baseline in days. Nominally 27.88 days for Sector 1. Shorter baselines indicate stars observed only during part of the sector due to momentum dumps, safe mode events, or CCD edge proximity.
- `duty_cycle_obs` — Fraction of the total sector timeline covered by actual observations. Values below 0.85 indicate significant gaps and may reduce BLS sensitivity for long-period signals.

### 2.7 Stage 5 — Phase-Fold and View Construction (make_views)

The phase-folded views are the primary inputs to the CNN branches of the fusion model. Both views are constructed by folding the cleaned, normalized light curve at the BLS period and binning into fixed-size arrays.
**BTJD Time Correction**
Catalog transit times (t0) are stored in BJD (Barycentric Julian Date). TESS timestamps use BTJD (BJD − 2,457,000). The offset is applied before folding:

```python
t0_btjd = catalog_t0_bjd - BTJD_OFFSET
```

**Phase Folding**
Each timestamp is converted to a phase value between −0.5 and +0.5, with phase 0 corresponding to mid-transit:

```python
phase = ((time - t0_btjd) / period + 0.5) % 1.0 - 0.5
```

All transit events stack coherently at phase 0. Out-of-transit flux distributes uniformly across the remaining phase range. This stacking is what makes periodic signals visible despite per-transit noise — a signal buried in noise at any individual transit becomes clear when N transits are averaged together.

**Global View (2000 bins)**
The full phase range [−0.5, +0.5] is divided into 2000 equal bins. The median flux within each bin is computed and stored as a single value:

```python
global_view = median_bin(phase, flux, n_bins=2000, phase_min=-0.5, phase_max=0.5)
```

2000 bins across a full orbit gives each bin a width of 0.05% of the period. For a 3-day period this is ~2 minutes per bin — approximately one TESS cadence. The global view captures:

- The primary transit dip at phase 0
- Any secondary eclipse at phase ±0.5
- Out-of-transit variability (ellipsoidal variations, reflection effects)
- The overall shape of the phase curve

**Local View (201 bins)**
The local view zooms into the ±`LOCAL*FRAC` × period window around phase 0, providing higher phase resolution around the transit itself:

```python
window_mask = np.abs(phase) < LOCAL_FRAC # ±10% of period
local_view = median_bin(phase[window_mask], flux[window_mask],
n_bins=201,
phase_min=-LOCAL_FRAC, phase_max=LOCAL_FRAC)
```

201 bins across 20% of the period gives each bin ~0.1% of the period. For the same 3-day example this is ~4.3 minutes per bin — roughly two TESS cadences. This resolution allows the CNN to learn ingress and egress shape separately, and to distinguish the flat bottom of a full transit from the pointed minimum of a V-shaped eclipse.

The asymmetry between 2000 global bins and 201 local bins is intentional and follows the AstroNet architecture: the global view prioritizes coverage of the full orbit while the local view prioritizes resolution of the transit shape.

**Median Binning Implementation**

```python
def median_bin(phase, flux, n_bins, phase_min, phase_max):
    edges = np.linspace(phase_min, phase_max, n_bins + 1)
    indices = np.digitize(phase, edges) - 1
    indices = np.clip(indices, 0, n_bins - 1)
    result = np.zeros(n_bins, dtype=np.float32)
    for i in range(n_bins):
        mask = indices == i
        result[i] = np.nanmedian(flux[mask]) if mask.any() else np.nan
    nans = np.isnan(result)
    if nans.any() and not nans.all():
        idx = np.arange(n_bins)
        result[nans] = np.interp(idx[nans], idx[~nans], result[~nans])
    elif nans.all():
        result[:] = 0.0
    return result
```

`np.digitize` vectorizes the bin assignment across all cadences in a single operation. Empty bins (no cadences in that phase range — common at the edges of the local view for long-period planets with few transits) are filled by linear interpolation between neighboring bins. If all bins are empty (pathological case) the view is zeroed.

### 2.8 Stage 6 — Augmentation (`augment_views`)

Data augmentation creates additional training samples by applying small physically-motivated perturbations to existing views. Two augmented copies are created per real sample (N_AUGMENTS = 2), tripling the effective dataset size.

```python
def augment_views(global_view, local_view, rng):
    augmented = []
    for _ in range(N_AUGMENTS):
        # Phase jitter: shift the light curve by a small random phase offset
        shift = int(rng.normal(0, PHASE_JITTER_STD * GLOBAL_BINS))
        g_aug = np.roll(global_view, shift)

        # Flux noise injection: add Gaussian noise scaled to the noise level
        noise_g = rng.normal(0, FLUX_NOISE_STD, size=global_view.shape)
        noise_l = rng.normal(0, FLUX_NOISE_STD, size=local_view.shape)
        g_aug = g_aug + noise_g.astype(np.float32)
        l_aug = local_view + noise_l.astype(np.float32)

        augmented.append((g_aug.astype(np.float32), l_aug.astype(np.float32)))
    return augmented
```

**Phase jitter** (`PHASE_JITTER_STD = 0.01`) shifts the global view by a random number of bins sampled from N(0, 0.01 × 2000) = N(0, 20) bins. This corresponds to a random phase offset of ±1% of the period — small enough not to misalign the transit, large enough to teach the model that the transit can appear at slightly different phase positions due to t0 uncertainty.

**Flux noise injection** (`FLUX_NOISE_STD = 0.002`) adds Gaussian noise at the 0.2% level to both views. This prevents overfitting to the exact noise realization in each light curve and teaches the model to recognize transit shapes across a range of SNR levels.

Augmentation is only applied in train mode and only to records with real labels (not −1). In science mode, all 5,196 records in the output include 1,732 real samples and 3,464 augmented copies of those — the augmentation still ran because BUILD_MODE = "train" was set with the understanding that the −1 labels are placeholder values, not meaningful class assignments.

## Part 3 — Output Files

- `final_tess_dataset.pkl`
  A pandas DataFrame with one row per sample (real + augmented). Contains all metadata columns plus the engineered scalar features. Does not contain the view arrays (those are in separate numpy files for memory efficiency).
  Shape: (5196, 31) — 1732 real + 3464 augmented rows.
  Key columns: `target_id, sector, mission, label, augmented, fold_period, fold_t0, fold_duration, plus all 23 scalar features.`

- `global_views.npy`
  Phase-folded global view arrays for all samples.
  Shape: (5196, 2000), dtype float32.
  Each row is the full-orbit phase curve of one star binned into 2000 points. The transit appears as a negative dip at the center of the array (index ~1000, corresponding to phase 0).

- `local_views.npy`
  Phase-folded local view arrays for all samples.
  Shape: (5196, 201), dtype float32.
  Each row is the zoomed transit window for one star. The transit dip appears at the center (index ~100).

- `scalar_features.npy`
  All engineered scalar features stacked into a 2D array.
  Shape: (5196, 23), dtype float32.
  Column order matches SCALAR_COLS in the builder. Two columns (secondary_depth, secondary_primary_ratio) have 0.1% NaN values which should be imputed (median fill) before model input. Two columns (odd_even_depth_ratio, baseline_kurtosis) have extreme outlier values and should be clipped (odd_even_depth_ratio at 10, baseline_kurtosis at 20) before model input.

- `labels.npy`
  Integer labels for all samples.
  Shape: (5196,), dtype int64.
  All values are −1 for this science dataset, indicating unlabeled inference targets. This file is retained for structural consistency with the training pipeline — downstream code loads the same set of five numpy files regardless of dataset mode.

- `missions.npy`
  Mission string for all samples.
  Shape: (5196,), dtype object.
  All values are "TESS" for this dataset.

## Part 4 — Known Issues and Mitigations

**BLS Power Stale Reference Bug**
The original BLS implementation used model.autopower() which returned a result object, but the code read best_period from a variable that was not updated on each call. This caused all 1,732 stars to return the same BLS period and SNR, producing a dataset where 99.5% of stars appeared to have "no signal."
Fix: Replaced autopower() with explicit model.power() on a fixed period grid, then extracted results using np.argmax(result.power) to index into the correct position in each result array.

**Windows Threading Stdout Crash**
print() calls inside ThreadPoolExecutor workers caused ValueError: I/O operation on closed file on Windows due to a race condition between the executor's shutdown and stdout buffer flushing.
Fix: All print statements inside threaded or multiprocessed functions are wrapped in a threading.Lock via safe_print() defined in config.py. The end='\r' carriage return style was also removed as it caused additional Windows console issues.

**Multiprocessing Safety**
An attempt to parallelize the build stage using ProcessPoolExecutor was abandoned because NumPy, SciPy, and Astropy internally use OpenBLAS thread pools that are not fork-safe on Windows. Spawning multiple processes caused silent data corruption and dropped records even with correct locking.
Resolution: The builder runs single-threaded. Speed is recovered by reducing the BLS period grid from 5,000 to 2,000 points and the duration grid from 20 to 4 points, reducing BLS compute time per star by approximately 6x with negligible accuracy loss.

**Augmentation Label Guard Conflict**
The augmentation block was guarded by int(row["label"]) != -1, designed to prevent augmentation of unlabeled science data. When running in train mode with a parquet file where labels were stored as −1 (because the collector used science mode defaults), the guard silently blocked all augmentation, producing a dataset where entries equaled processed count rather than 3x processed count.

Fix: The label guard was removed from the augmentation condition. Mode control is now handled entirely by BUILD_MODE at the top of the file.

**odd_even_depth_ratio Overflow**
Stars with near-zero mean transit depth produce astronomically large odd-even ratio values (maximum observed: 448.5) due to division by a near-zero denominator. This does not indicate a real signal — it is numerical noise.
Mitigation: Clip to 10 before any model input. The physically meaningful range is 0-2; values above 10 carry no additional information.

## Part 5 — Dataset Statistics (Sector 1)

| Metric                         | Value                                 |
| ------------------------------ | ------------------------------------- |
| Total stars downloaded         | 1732                                  |
| Stars Skipped                  | 0                                     |
| Sector                         | 1                                     |
| Observation Baseline           | 27.88 Days                            |
| Median Cadences per star       | 18,274                                |
| Median flux error (normalized) | 0.0010                                |
| BLS SNR Median                 | 6.59                                  |
| BLS SNR max                    | 290.66                                |
| Stars with BLS SNR > 6         | ~7                                    |
| Stars flagged as likely EB     | ~15% by secondary/primary ratio > 0.1 |
| Scalar feature NaN rate        | <0.2% (all features)                  |
| Augmented Dataset Size         | 5,196 (1,732 real + 3,464 augmented)  |
| Global View Shape              | (5196,2000)                           |
| Local View Shape               | (5196, 201)                           |
| Scalar Features                | 23 columns                            |
