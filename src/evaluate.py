"""Model evaluation utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.metrics import (
    brier_score_loss,
    precision_recall_curve,
    auc,
    roc_auc_score,
    roc_curve,
)

from src.config import FIGURES, METRICS_FILE


def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Kolmogorov–Smirnov statistic (max separation of CDFs)."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def gini(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Gini coefficient = 2 * ROC-AUC - 1."""
    return 2.0 * roc_auc_score(y_true, y_prob) - 1.0


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    """Return roc_auc, pr_auc, ks, gini, brier for a binary classifier."""
    roc_auc = float(roc_auc_score(y_true, y_prob))
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc_val = float(auc(recall, precision))
    ks = ks_statistic(y_true, y_prob)
    gini_val = gini(y_true, y_prob)
    brier = float(brier_score_loss(y_true, y_prob))
    return {
        "roc_auc": round(roc_auc, 6),
        "pr_auc": round(pr_auc_val, 6),
        "ks": round(ks, 6),
        "gini": round(gini_val, 6),
        "brier": round(brier, 6),
    }


def save_metrics(key: str, data: dict[str, Any]) -> None:
    """Merge *data* into reports/metrics.json under *key* without overwriting others."""
    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if METRICS_FILE.exists():
        with open(METRICS_FILE, "r", encoding="utf-8") as f:
            existing = json.load(f)
    existing[key] = data
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)


def calibrate_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[Any, dict[str, Any]]:
    """Calibrate *model* using both isotonic and sigmoid methods.

    Chooses the method with lower Brier score on a held-out validation split
    (carved from training data only — test set never used for selection).
    Saves a calibration curve figure and stores results in metrics.json.

    Calibration matters because cost-based threshold optimisation assumes
    probabilities are well-calibrated; a model that outputs 0.9 should default
    90% of the time, otherwise the cost calculation is systematically biased.
    """
    from sklearn.model_selection import train_test_split
    from src.config import SEED

    # Split training data for calibration model selection
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=SEED, stratify=y_train
    )

    brier_before = float(brier_score_loss(y_test, model.predict_proba(X_test)[:, 1]))

    best_brier = brier_before
    best_method = "none"
    best_model = model

    for method in ["isotonic", "sigmoid"]:
        cal = CalibratedClassifierCV(model, method=method, cv="prefit")
        cal.fit(X_val, y_val)
        b = float(brier_score_loss(y_val, cal.predict_proba(X_val)[:, 1]))
        if b < best_brier:
            best_brier = b
            best_method = method
            best_model = cal

    brier_after = float(
        brier_score_loss(y_test, best_model.predict_proba(X_test)[:, 1])
    )

    # Calibration plot
    FIGURES.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    CalibrationDisplay.from_predictions(
        y_test, model.predict_proba(X_test)[:, 1],
        n_bins=10, ax=ax, label="Before calibration", color="steelblue"
    )
    if best_method != "none":
        CalibrationDisplay.from_predictions(
            y_test, best_model.predict_proba(X_test)[:, 1],
            n_bins=10, ax=ax, label=f"After ({best_method})", color="tomato"
        )
    ax.set_title("Calibration Curve")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "calibration.png", dpi=100)
    plt.close()

    result = {
        "brier_before": round(brier_before, 6),
        "brier_after": round(brier_after, 6),
        "method": best_method,
        "improved": brier_after < brier_before,
    }
    save_metrics("calibration", result)
    print(f"Calibration: {best_method}, Brier before={brier_before:.4f}, after={brier_after:.4f}")
    return best_model, result
