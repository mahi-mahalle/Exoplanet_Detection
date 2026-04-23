# src/ml_model.py

import numpy as np
import pandas as pd
import joblib
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from config import TEST_SIZE, RANDOM_STATE, N_AUGMENTATIONS, MODEL_PATH, OUTPUT_DIR

FEATURE_NAMES = [
    "depth", "duration", "mean_flux_in",
    "std_flux_in", "std_flux_out", "snr",
    "symmetry", "sharpness", "n_transits"
]

def build_dataset(true_features, lc_flat, bls_output):
    """
    Generates synthetic positive and negative samples
    via perturbation and wrong-period folding.
    """
    from src.analyzer import analyze_transit
    import copy

    samples = []
    labels = []

    # --- Positive samples (label=1) ---
    # Augment true transit features with Gaussian noise
    for _ in range(N_AUGMENTATIONS):
        perturbed = {}
        for k, v in true_features.items():
            noise = np.random.normal(0, abs(v) * 0.05 + 1e-6)
            perturbed[k] = v + noise
        samples.append([perturbed[f] for f in FEATURE_NAMES])
        labels.append(1)

    # --- Negative samples (label=0) ---
    # Fold at wrong periods → no coherent dip → extract features
    wrong_periods = [
        bls_output["best_period"] * 2,
        bls_output["best_period"] * 3,
        bls_output["best_period"] * 0.5,
        bls_output["best_period"] * 0.33,
        1.0, 2.0, 5.0, 7.0, 10.0, 13.0
    ]

    for wp in wrong_periods:
        try:
            fake_bls = copy.deepcopy(bls_output)
            fake_bls["best_period"] = wp
            fake_bls["t0"] = bls_output["t0"]
            fake_bls["duration"] = bls_output["duration"]
            _, neg_features, _ = analyze_transit(lc_flat, fake_bls)

            for _ in range(N_AUGMENTATIONS // len(wrong_periods)):
                perturbed = {}
                for k, v in neg_features.items():
                    noise = np.random.normal(0, abs(v) * 0.05 + 1e-6)
                    perturbed[k] = v + noise
                samples.append([perturbed[f] for f in FEATURE_NAMES])
                labels.append(0)
        except:
            continue

    return np.array(samples), np.array(labels)


def train_model(X, y):
    """
    Trains XGBoost classifier and saves model + evaluation plots.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=RANDOM_STATE
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    y_pred = model.predict(X_test)
    print("\n[ML] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Noise", "Transit"]))

    # Save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Noise", "Transit"])
    fig, ax = plt.subplots()
    disp.plot(ax=ax, colorbar=False)
    plt.title("XGBoost Confusion Matrix")
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), bbox_inches="tight")
    plt.close()

    # Save feature importance
    importance = model.feature_importances_
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(FEATURE_NAMES, importance, color="steelblue")
    ax.set_xlabel("Importance (Gain)")
    ax.set_title("XGBoost Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), bbox_inches="tight")
    plt.close()

    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"[ML] Model saved to {MODEL_PATH}")

    return model


def predict(features_dict):
    """
    Loads saved model and predicts on a single feature vector.
    Returns label and confidence score.
    """
    model = joblib.load(MODEL_PATH)
    X = np.array([[features_dict[f] for f in FEATURE_NAMES]])
    label = model.predict(X)[0]
    confidence = model.predict_proba(X)[0][label]
    return int(label), round(float(confidence) * 100, 2)