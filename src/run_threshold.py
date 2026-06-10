"""Cost-based threshold optimisation."""
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
from sklearn.model_selection import StratifiedKFold

from src.config import COST_FN, COST_FP, DATA_PROCESSED, FIGURES, SEED, SENSITIVITY_FILE
from src.data import download_data, load_clean, split_data
from src.features import build_features
from src.models import train_lgbm
from src.evaluate import calibrate_model, save_metrics
from src.threshold import expected_loss, optimize_threshold, sensitivity_table


def main() -> None:
    """Optimise decision threshold using out-of-fold predictions on training data."""
    download_data()
    df = load_clean()
    df = build_features(df)
    X_train, X_test, y_train, y_test = split_data(df)

    model = train_lgbm(X_train, y_train)

    # Calibrate on training data only — threshold chosen on OOF, never test set
    calibrated_model, _ = calibrate_model(model, X_train, y_train, X_test, y_test)

    # Out-of-fold predictions on training data for threshold selection
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_probs = np.zeros(len(y_train))
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train = y_train.iloc[train_idx]
        fold_model = train_lgbm(X_fold_train, y_fold_train)
        oof_probs[val_idx] = fold_model.predict_proba(X_fold_val)[:, 1]

    # Choose threshold on OOF predictions
    opt_threshold, oof_loss = optimize_threshold(y_train.values, oof_probs)
    print(f"\nOptimal threshold (from OOF): {opt_threshold:.4f}")
    print(f"OOF expected loss at optimal: {oof_loss:.4f}")

    # Evaluate fixed threshold on test set
    y_prob_test = calibrated_model.predict_proba(X_test)[:, 1]
    loss_at_05 = expected_loss(y_test.values, y_prob_test, 0.5)
    loss_at_opt = expected_loss(y_test.values, y_prob_test, opt_threshold)
    loss_reduction = (1 - loss_at_opt / loss_at_05) * 100 if loss_at_05 > 0 else 0.0

    print(f"Test expected loss @ 0.5:    {loss_at_05:.4f}")
    print(f"Test expected loss @ {opt_threshold:.2f}: {loss_at_opt:.4f}")
    print(f"Loss reduction vs 0.5:        {loss_reduction:.1f}%")

    # Decile lift analysis
    n_test = len(y_test)
    top_decile_n = n_test // 10
    sorted_idx = np.argsort(y_prob_test)[::-1]
    top_decile_idx = sorted_idx[:top_decile_n]
    top_decile_defaults = y_test.values[top_decile_idx].sum()
    total_defaults = y_test.values.sum()
    top_decile_capture = float(top_decile_defaults / total_defaults)
    top_decile_default_rate = float(top_decile_defaults / top_decile_n)
    overall_default_rate = float(y_test.values.mean())
    top_decile_lift = top_decile_default_rate / overall_default_rate

    print(f"\nTop decile capture: {top_decile_capture:.4f}")
    print(f"Top decile lift:    {top_decile_lift:.4f}")

    # Sensitivity table
    sens_df = sensitivity_table(y_test.values, y_prob_test)
    SENSITIVITY_FILE.parent.mkdir(parents=True, exist_ok=True)
    sens_df.to_csv(SENSITIVITY_FILE, index=False)
    print(f"\nSensitivity table saved to {SENSITIVITY_FILE}")
    print(sens_df.to_string(index=False))

    # Threshold plot
    FIGURES.mkdir(parents=True, exist_ok=True)
    thresholds = np.arange(0.01, 1.0, 0.01)
    losses = [expected_loss(y_test.values, y_prob_test, t) for t in thresholds]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, losses, label="Expected Loss", color="steelblue")
    ax.axvline(0.5, color="gray", linestyle="--", label=f"Threshold 0.5 (loss={loss_at_05:.3f})")
    ax.axvline(opt_threshold, color="tomato", linestyle="-",
               label=f"Optimal {opt_threshold:.2f} (loss={loss_at_opt:.3f})")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Expected Loss per Account")
    ax.set_title("Expected Loss vs Decision Threshold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "threshold.png", dpi=100)
    plt.close()
    print(f"Saved threshold plot to {FIGURES / 'threshold.png'}")

    # Save to metrics.json
    save_metrics("business", {
        "optimal_threshold": round(opt_threshold, 4),
        "loss_at_0.5": round(loss_at_05, 6),
        "loss_at_optimal": round(loss_at_opt, 6),
        "loss_reduction_pct": round(loss_reduction, 2),
        "top_decile_capture": round(top_decile_capture, 6),
        "top_decile_lift": round(top_decile_lift, 4),
    })


if __name__ == "__main__":
    main()
