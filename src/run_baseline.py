"""Train and evaluate the logistic regression baseline."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np

from src.config import SEED
from src.data import download_data, load_clean, split_data
from src.features import build_features
from src.models import train_logistic
from src.evaluate import compute_metrics, save_metrics


def main() -> None:
    """Train baseline logistic regression and print/save metrics."""
    download_data()
    df = load_clean()
    df = build_features(df)
    X_train, X_test, y_train, y_test = split_data(df)

    model = train_logistic(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test.values, y_prob)
    print("\n=== Baseline Test Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # 5-fold CV on training data
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"\nCV ROC-AUC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    result = {
        **metrics,
        "cv_roc_auc_mean": round(float(cv_scores.mean()), 6),
        "cv_roc_auc_std": round(float(cv_scores.std()), 6),
    }

    save_metrics("baseline", result)
    print("\nSaved to reports/metrics.json under 'baseline'")

    # Leakage guard
    if metrics["roc_auc"] > 0.85:
        print("WARNING: ROC-AUC > 0.85 on test — possible leakage!")


if __name__ == "__main__":
    main()
