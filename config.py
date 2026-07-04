"""config.py — Shared configuration and utility functions.

All collector scripts (collect_kepler.py, collect_k2.py, collect_tess.py)
and the pipeline (build_dataset.py) import from here.  Change values in
ONE place and they propagate everywhere.
"""

import sys
import threading
import warnings

import numpy as np

warnings.filterwarnings("ignore")

# ── RUN PARAMETERS ────────────────────────────────────────────────────────
N_SAMPLES        = 10          # targets per mission; raise to 100/500/1000 when ready
MAX_WORKERS      = 6           # parallel download threads (MAST rate-limits > 4–6)
STAR_TIMEOUT     = 60          # seconds the main thread waits for one star's future
CHECKPOINT_EVERY = 25          # pickle checkpoint cadence (records, not stars)
MAX_KEPLER_QTRS  = 2           # Kepler quarters to download per star

# ── BLS / FOLD PARAMETERS ─────────────────────────────────────────────────
PERIOD_MIN  = 0.5              # days
PERIOD_MAX  = 15.0             # days
GLOBAL_BINS = 2000             # phase-fold bins for global view
LOCAL_BINS  = 200              # phase-fold bins for local (transit) view
LOCAL_FRAC  = 0.1              # ±fraction of period shown in local view

# ── PATHS ─────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = "checkpoints"
RAW_DATA_DIR   = "raw_data"    # collector outputs land here (parquet files)
SAVE_PATH      = "curated_dataset.pkl"

# ── LABEL MAP ─────────────────────────────────────────────────────────────
LABEL_MAP = {
    # Standard long-form (TESS / Kepler)
    "CONFIRMED":                   0,
    "CANDIDATE":                   1,
    "FALSE POSITIVE":              2,
    # Kepler KOI short codes
    "KP": 0, "CP": 0,
    "PC": 1,
    "FP": 2, "EB": 2, "FA": 2,
    # Longer forms seen in various archive tables
    "CONFIRMED PLANET":            0,
    "PLANET CANDIDATE":            1,
    # k2pandc-specific strings
    "FALSE POSITIVE [CANDIDATE]":  2,
    "REFUTED [PLANET]":            2,
}

# ── THREAD-SAFE PRINT ─────────────────────────────────────────────────────
_print_lock = threading.Lock()

def safe_print(*args, **kwargs):
    """Thread-safe print that never raises if the underlying stream is
    in a bad state (Windows console teardown / Ctrl+C race).

    Mirrors the safe_print() in the original DataBuilder.py but lives
    here so every script shares exactly one implementation.
    """
    with _print_lock:
        try:
            print(*args, **kwargs)
        except ValueError:
            try:
                print(*args, file=sys.stderr, **kwargs)
            except Exception:
                pass
        except Exception:
            pass

# ── SMALL DATA HELPERS ────────────────────────────────────────────────────

def safe_float(val):
    """Convert val to float; return np.nan on failure or non-finite."""
    try:
        v = float(val)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def normalize_flux(flux):
    """Median-MAD normalisation → roughly zero-centred, unit-scaled."""
    flux   = np.array(flux, dtype=np.float32)
    median = np.nanmedian(flux)
    mad    = np.nanmedian(np.abs(flux - median))
    if mad == 0 or np.isnan(mad):
        return flux - median
    return (flux - median) / mad


def pad_or_trim(arr, length):
    """Return a float32 array of exactly `length` elements."""
    if arr is None:
        return np.zeros(length, dtype=np.float32)
    arr = np.array(arr, dtype=np.float32)
    if len(arr) >= length:
        return arr[:length]
    return np.pad(arr, (0, length - len(arr)), constant_values=0.0)


def stratified_sample(df, n, label_col="label", random_state=42):
    """Sample up to n rows from df, balanced across label_col.

    Avoids the ValueError raised by the original group.sample() when a
    label class has 0 rows (empty group crash with min-size=1 guard).
    """
    import pandas as pd

    if df.empty:
        return df

    per_label = max(n // 3, 1)
    parts = []
    for _, group in df.groupby(label_col, group_keys=False):
        if len(group) == 0:
            continue
        take = min(len(group), per_label)
        parts.append(group.sample(take, random_state=random_state))

    if not parts:
        return df.iloc[0:0]

    sampled = pd.concat(parts, ignore_index=False)
    return (
        sampled.sample(min(n, len(sampled)), random_state=random_state)
               .reset_index(drop=True)
    )
