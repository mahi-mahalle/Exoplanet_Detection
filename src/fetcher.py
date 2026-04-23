# src/fetcher.py

import lightkurve as lk
import os
import pickle
from config import MISSION, DATA_DIR

def fetch_lightcurve(star_name):
    """
    Downloads and stitches all available TESS sectors for a given star.
    Caches result locally to avoid re-downloading.
    """
    cache_file = os.path.join(DATA_DIR, f"{star_name.replace(' ', '_')}_lc.pkl")

    if os.path.exists(cache_file):
        print(f"[Fetcher] Loading cached data for {star_name}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print(f"[Fetcher] Searching {MISSION} for {star_name}...")
    search = lk.search_lightcurve(star_name, mission=MISSION)

    if len(search) == 0:
        raise ValueError(f"No data found for {star_name} in {MISSION}")

    print(f"[Fetcher] Found {len(search)} sector(s). Downloading all...")
    lc_collection = search.download_all()

    # Stitch all sectors into one light curve
    lc = lc_collection.stitch().remove_nans()

    # Apply TESS quality flags
    lc = lc[lc.quality == 0]

    print(f"[Fetcher] Done. Total cadences: {len(lc)}")

    # Cache it
    with open(cache_file, "wb") as f:
        pickle.dump(lc, f)

    return lc