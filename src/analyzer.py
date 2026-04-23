# src/analyzer.py

import numpy as np
import astropy.constants as C
from config import PHASE_WIDTH, STAR_RADIUS_SOLAR, PLANET_RADIUS_JUP, STAR_MASS_SOLAR, STAR_TEFF

def analyze_transit(lc_flat, bls_output):
    """
    Phase-folds, measures transit depth, Rp/Rs,
    and extracts ML features from the detected transit.
    """
    best_period = bls_output["best_period"]
    t0 = bls_output["t0"]
    duration = bls_output["duration"]

    # Phase fold
    folded = lc_flat.fold(period=best_period, epoch_time=t0)
    phase = folded.phase.value
    flux = folded.flux.value

    # In/out transit masks
    in_transit = np.abs(phase) < PHASE_WIDTH
    out_transit = ~in_transit

    f_in = np.median(flux[in_transit])
    f_out = np.median(flux[out_transit])

    # Physical parameters
    depth = (f_out - f_in) / f_out
    rp_over_rs = np.sqrt(max(depth, 0))

    star_radius = STAR_RADIUS_SOLAR * C.R_sun
    planet_radius_ref = PLANET_RADIUS_JUP * C.R_jup
    rp_over_rs_literature = (planet_radius_ref / star_radius).decompose().value

    error_percent = abs(rp_over_rs - rp_over_rs_literature) / rp_over_rs_literature * 100

    # Semi-major axis via Kepler's 3rd Law
    G = C.G.value
    M_star = STAR_MASS_SOLAR * C.M_sun.value
    P_sec = best_period * 86400
    a_meters = (G * M_star * P_sec**2 / (4 * np.pi**2)) ** (1/3)
    a_AU = a_meters / C.au.value

    # Equilibrium temperature
    albedo = 0.3
    T_eq = STAR_TEFF * (1 - albedo)**0.25 * np.sqrt(STAR_RADIUS_SOLAR * C.R_sun.value / (2 * a_meters))

    # ML Feature Extraction
    std_in = np.std(flux[in_transit])
    std_out = np.std(flux[out_transit])
    snr = depth / std_out if std_out > 0 else 0

    # Symmetry: compare left vs right half of transit dip
    left = flux[(phase > -PHASE_WIDTH) & (phase < 0)]
    right = flux[(phase > 0) & (phase < PHASE_WIDTH)]
    min_len = min(len(left), len(right))
    if min_len > 0:
        symmetry = 1 - np.mean(np.abs(left[:min_len] - right[:min_len][::-1]))
    else:
        symmetry = 0

    n_transits = int((lc_flat.time.value[-1] - lc_flat.time.value[0]) / best_period)

    features = {
        "depth": depth,
        "duration": duration,
        "mean_flux_in": f_in,
        "std_flux_in": std_in,
        "std_flux_out": std_out,
        "snr": snr,
        "symmetry": symmetry,
        "sharpness": depth / duration if duration > 0 else 0,
        "n_transits": n_transits
    }

    physical = {
        "period": best_period,
        "depth": depth,
        "rp_over_rs": rp_over_rs,
        "rp_over_rs_literature": rp_over_rs_literature,
        "error_percent": error_percent,
        "semi_major_axis_AU": a_AU,
        "eq_temperature_K": T_eq
    }

    return folded, features, physical