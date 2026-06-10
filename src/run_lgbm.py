"""Train and evaluate the LightGBM model."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_curve, roc_auc_score

from src.config import DATA_PROCESSED, FIGURES, SEED
from src.data import download_data, load_clean, split_data
from src.features import build_features
from src.models import train_logistic, train_lgbm
from src.evaluate import compute_metrics, save_metrics, calibrate_model


def roc_auc_score_val(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute ROC-AUC score."""
    return float(roc_auc_score(y_true, y_prob))


def main() -> None:
    """Train LightGBM, compare to baseline, calibrate, save predictions and ROC figure."""
    download_data()
    df = load_clean()
    df = build_features(df)
    X_train, X_test, y_train, y_test = split_data(df)

    # LightGBM
    lgbm_model = train_lgbm(X_train, y_train)
    y_prob_lgbm = lgbm_model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test.values, y_prob_lgbm)
    print("\n=== LightGBM Test Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(lgbm_model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"\nCV ROC-AUC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    result = {
        **metrics,
        "cv_roc_auc_mean": round(float(cv_scores.mean()), 6),
        "cv_roc_auc_std": round(float(cv_scores.std()), 6),
    }
    save_metrics("lgbm", result)

    # Save predictions
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    preds_df = pd.DataFrame({
        "y_true": y_test.values,
        "y_prob": y_prob_lgbm,
    })
    preds_df.to_csv(DATA_PROCESSED / "test_preds.csv", index=False)
    print(f"Saved test predictions to {DATA_PROCESSED / 'test_preds.csv'}")

    # ROC curve comparison
    baseline = train_logistic(X_train, y_train)
    y_prob_baseline = baseline.predict_proba(X_test)[:, 1]

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    for y_prob, label, color in [
        (y_prob_baseline, f"Baseline LR (AUC={roc_auc_score_val(y_test.values, y_prob_baseline):.3f})", "steelblue"),
        (y_prob_lgbm, f"LightGBM (AUC={roc_auc_score_val(y_test.values, y_prob_lgbm):.3f})", "tomato"),
    ]:
        fpr, tpr, _ = roc_curve(y_test.values, y_prob)
        ax.plot(fpr, tpr, label=label, color=color)
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "roc.png", dpi=100)
    plt.close()
    print(f"Saved ROC curve to {FIGURES / 'roc.png'}")

    # Calibration
    calibrated_model, cal_result = calibrate_model(
        lgbm_model, X_train, y_train, X_test, y_test
    )
    print(f"Calibration result: {cal_result}")

    if metrics["roc_auc"] > 0.85:
        print("WARNING: LightGBM ROC-AUC > 0.85 — investigate leakage!")


if __name__ == "__main__":
    main()
