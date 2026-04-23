# src/detector.py

import numpy as np
from astropy.timeseries import BoxLeastSquares
from config import PERIOD_MIN, PERIOD_MAX, PERIOD_STEPS, TRANSIT_DURATIONS

def run_bls(lc_flat):
    """
    Runs Box Least Squares over multiple transit duration guesses.
    Returns best period, t0, duration, and full BLS result object.
    """
    t = lc_flat.time.value
    f = lc_flat.flux.value

    periods = np.linspace(PERIOD_MIN, PERIOD_MAX, PERIOD_STEPS)

    best_power = -np.inf
    best_result = None

    for duration in TRANSIT_DURATIONS:
        bls = BoxLeastSquares(t, f)
        result = bls.power(periods, duration)
        peak = np.max(result.power)

        print(f"[Detector] Duration {duration}d → Peak power: {peak:.4f}")

        if peak > best_power:
            best_power = peak
            best_result = result

    # Extract best parameters
    idx = np.argmax(best_result.power)
    best_period = best_result.period[idx]
    t0 = best_result.transit_time[idx]
    duration = best_result.duration[idx]

    # Top 3 candidate periods (for display)
    top3_idx = np.argsort(best_result.power)[-3:][::-1]
    top3_periods = best_result.period[top3_idx]
    top3_powers = best_result.power[top3_idx]

    print(f"[Detector] Best period: {best_period:.5f} days")
    print(f"[Detector] Transit time t0: {t0:.4f} BTJD")
    print(f"[Detector] Transit duration: {duration:.4f} days")

    return {
        "best_period": best_period,
        "t0": t0,
        "duration": duration,
        "bls_result": best_result,
        "periods": periods,
        "top3_periods": top3_periods,
        "top3_powers": top3_powers
    }