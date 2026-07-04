"""collect_science.py — Download one full TESS sector from MAST bulk.
No labels. This is the unlabeled science dataset.
"""

import os, time, pickle
import numpy as np
import pandas as pd
import lightkurve as lk
from astroquery.mast import Observations
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import MAX_WORKERS, STAR_TIMEOUT, CHECKPOINT_EVERY, safe_print, safe_float

SECTOR    = 1          # change to whichever sector you want
N_LIMIT   = None       # None = download all ~20k, set to 500 for testing
RAW_OUT   = f"raw_data/science_sector{SECTOR}.parquet"
CKPT_PATH = f"checkpoints/science_sector{SECTOR}.pkl"

# ── STEP 1: Get all TIC IDs observed at 2-min cadence in this sector ──────

def fetch_sector_targets(sector: int) -> pd.DataFrame:
    safe_print(f"  Querying MAST for all 2-min targets in Sector {sector}...")
    obs = Observations.query_criteria(
        obs_collection = "TESS",
        dataproduct_type = "timeseries",
        sequence_number  = sector,
        t_exptime        = [100, 140],   # 2-min cadence = 120s ± buffer
    )
    df = obs.to_pandas()
    # target_name column contains the TIC ID as "TIC XXXXXXXXX"
    df["tic_id"] = (df["target_name"]
                    .str.replace("TIC ", "", regex=False)
                    .str.strip())
    df["tic_id"] = pd.to_numeric(df["tic_id"], errors="coerce")
    df = df.dropna(subset=["tic_id"])
    df["tic_id"] = df["tic_id"].astype(int)
    safe_print(f"  Found {len(df)} targets in Sector {sector}")
    return df[["tic_id"]].drop_duplicates().reset_index(drop=True)

# ── STEP 2: Download each light curve ────────────────────────────────────

def download_one(tic_id: int, sector: int) -> dict | None:
    try:
        results = lk.search_lightcurve(
            f"TIC {tic_id}", mission="TESS",
            exptime=120, sector=sector
        )
        if results is None or len(results) == 0:
            return None

        lc = results[0].download()
        if lc is None or len(lc.time) < 100:
            return None

        lc = lc.remove_nans().remove_outliers().normalize()

        return {
            "target_id": tic_id,
            "sector":    sector,
            "mission":   "TESS",
            "label":     -1,           # -1 = unlabeled science target
            "time":      np.asarray(lc.time.value,  dtype=np.float64),
            "flux":      np.asarray(lc.flux.value,   dtype=np.float32),
            "flux_err":  np.asarray(lc.flux_err.value, dtype=np.float32),
        }
    except Exception as exc:
        safe_print(f"  [SKIP] TIC {tic_id} — {type(exc).__name__}: {exc}")
        return None

# ── STEP 3: Parallel runner with checkpointing ────────────────────────────

def collect_science(sector: int = SECTOR,
                    n_limit: int = None,
                    fresh: bool = False) -> list[dict]:

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("raw_data",    exist_ok=True)

    if fresh and os.path.exists(CKPT_PATH):
        os.remove(CKPT_PATH)

    records: list[dict] = []
    if os.path.exists(CKPT_PATH):
        with open(CKPT_PATH, "rb") as fh:
            records = pickle.load(fh)
        safe_print(f"  Resuming — {len(records)} records already downloaded.")

    targets   = fetch_sector_targets(sector)
    done_ids  = {r["target_id"] for r in records}
    remaining = targets[~targets["tic_id"].isin(done_ids)]["tic_id"].tolist()

    if n_limit is not None:
        remaining = remaining[:n_limit - len(records)]

    safe_print(f"  {len(remaining)} stars remaining to download.")

    t_start = time.time()
    completed = failed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {
            executor.submit(download_one, tic_id, sector): tic_id
            for tic_id in remaining
        }

        for future in as_completed(future_map):
            try:
                result = future.result(timeout=STAR_TIMEOUT)
            except Exception:
                failed += 1
                continue

            completed += 1
            if result is not None:
                records.append(result)

            if completed % 10 == 0:
                elapsed = time.time() - t_start
                rate    = completed / elapsed if elapsed > 0 else 0
                safe_print(
                    f"  {len(records)} collected | "
                    f"{completed} processed | {failed} failed | "
                    f"{rate:.2f} stars/s"
                )

            if records and len(records) % CHECKPOINT_EVERY == 0:
                with open(CKPT_PATH, "wb") as fh:
                    pickle.dump(records, fh)

    with open(CKPT_PATH, "wb") as fh:
        pickle.dump(records, fh)

    safe_print(f"\n  Done — {len(records)} science targets downloaded.")
    return records

# ── STEP 4: Save ──────────────────────────────────────────────────────────

def save(records: list[dict]) -> str:
    df = pd.DataFrame(records)
    df.to_parquet(RAW_OUT, index=False)
    safe_print(f"  Saved → {RAW_OUT}")
    return RAW_OUT

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--sector", type=int,  default=SECTOR)
    p.add_argument("--n",      type=int,  default=None)
    p.add_argument("--fresh",  action="store_true")
    args = p.parse_args()

    records = collect_science(args.sector, args.n, args.fresh)
    if records:
        save(records)