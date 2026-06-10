"""SHAP explainability for the LightGBM model."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURES, SEED
from src.evaluate import save_metrics


def compute_shap_explanations(
    model: object,
    X_test: pd.DataFrame,
    sample_size: int = 2000,
) -> None:
    """Compute SHAP values for LightGBM model on a sample of test data.

    Saves summary plot and top 5 features to metrics.json.
    Runs in under 1 minute.
    """
    import shap

    rng = np.random.default_rng(SEED)
    n = min(sample_size, len(X_test))
    idx = rng.choice(len(X_test), size=n, replace=False)
    X_sample = X_test.iloc[idx].reset_index(drop=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # For binary classification, shap_values may be a list [neg_class, pos_class]
    if isinstance(shap_values, list):
        sv = shap_values[1]  # positive class
    elif hasattr(shap_values, "values"):
        sv = shap_values.values
    else:
        sv = shap_values

    mean_abs_shap = np.abs(sv).mean(axis=0)
    top5_idx = np.argsort(mean_abs_shap)[::-1][:5]
    top5 = [
        {"feature": str(X_sample.columns[i]), "mean_abs_shap": round(float(mean_abs_shap[i]), 6)}
        for i in top5_idx
    ]

    print("\nTop 5 SHAP drivers:")
    for item in top5:
        print(f"  {item['feature']}: {item['mean_abs_shap']:.4f}")

    save_metrics("top_drivers", top5)

    # Summary plot
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(sv, X_sample, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(FIGURES / "shap_summary.png", dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP summary plot to {FIGURES / 'shap_summary.png'}")
