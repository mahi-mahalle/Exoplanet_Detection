# src/preprocessor.py

import numpy as np
from astropy.stats import sigma_clip as astropy_sigma_clip
from config import WINDOW_LENGTH, SIGMA_CLIP

def preprocess(lc):
    """
    Cleans and flattens a raw light curve.
    Returns both raw normalized and cleaned flattened versions.
    """
    # Step 1: Normalize flux to median = 1
    lc_normalized = lc.normalize()

    # Step 2: Sigma clip — remove cosmic rays / outliers
    flux_vals = lc_normalized.flux.value
    clipped = astropy_sigma_clip(flux_vals, sigma=SIGMA_CLIP, maxiters=5)
    mask = ~clipped.mask
    lc_clipped = lc_normalized[mask]

    # Step 3: Flatten — remove long-term stellar variability
    lc_flat, trend = lc_clipped.flatten(
        window_length=WINDOW_LENGTH,
        return_trend=True
    )

    print(f"[Preprocessor] Removed {np.sum(~mask)} outlier cadences")
    print(f"[Preprocessor] Remaining cadences: {len(lc_flat)}")

    return lc_normalized, lc_flat, trend