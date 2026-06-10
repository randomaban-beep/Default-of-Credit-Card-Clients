"""Run the complete pipeline end to end."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    """Execute all pipeline stages in order."""
    t0 = time.time()
    print("=" * 60)
    print("CREDIT DEFAULT PREDICTION PIPELINE")
    print("=" * 60)

    # Stage 1: Data
    print("\n[1/6] Data loading...")
    from src.data import download_data, load_clean, split_data
    from src.features import build_features
    download_data()
    df = load_clean()
    df = build_features(df)
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    # Stage 2: Baseline
    print("\n[2/6] Training baseline (LogisticRegression)...")
    from src.models import train_logistic
    from src.evaluate import compute_metrics, save_metrics
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from src.config import SEED
    import numpy as np

    baseline = train_logistic(X_train, y_train)
    y_prob_bl = baseline.predict_proba(X_test)[:, 1]
    bl_metrics = compute_metrics(y_test.values, y_prob_bl)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    bl_cv = cross_val_score(baseline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    save_metrics("baseline", {
        **bl_metrics,
        "cv_roc_auc_mean": round(float(bl_cv.mean()), 6),
        "cv_roc_auc_std": round(float(bl_cv.std()), 6),
    })
    print(f"  Baseline ROC-AUC: {bl_metrics['roc_auc']:.4f}")

    # Stage 3: LightGBM
    print("\n[3/6] Training LightGBM...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    from src.config import DATA_PROCESSED, FIGURES
    import pandas as pd

    from src.models import train_lgbm
    lgbm = train_lgbm(X_train, y_train)
    y_prob_lgbm = lgbm.predict_proba(X_test)[:, 1]
    lgbm_metrics = compute_metrics(y_test.values, y_prob_lgbm)
    lgbm_cv = cross_val_score(lgbm, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    save_metrics("lgbm", {
        **lgbm_metrics,
        "cv_roc_auc_mean": round(float(lgbm_cv.mean()), 6),
        "cv_roc_auc_std": round(float(lgbm_cv.std()), 6),
    })
    print(f"  LightGBM ROC-AUC: {lgbm_metrics['roc_auc']:.4f}")

    # Save predictions
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"y_true": y_test.values, "y_prob": y_prob_lgbm}).to_csv(
        DATA_PROCESSED / "test_preds.csv", index=False
    )

    # ROC curve
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    from sklearn.metrics import roc_auc_score
    for yp, lbl, col in [
        (y_prob_bl, f"LR (AUC={roc_auc_score(y_test.values, y_prob_bl):.3f})", "steelblue"),
        (y_prob_lgbm, f"LGBM (AUC={roc_auc_score(y_test.values, y_prob_lgbm):.3f})", "tomato"),
    ]:
        fpr, tpr, _ = roc_curve(y_test.values, yp)
        ax.plot(fpr, tpr, label=lbl, color=col)
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC Curves")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "roc.png", dpi=100)
    plt.close()

    # Stage 4: Calibration
    print("\n[4/6] Calibrating probabilities...")
    from src.evaluate import calibrate_model
    calibrated_model, cal_result = calibrate_model(lgbm, X_train, y_train, X_test, y_test)
    y_prob_cal = calibrated_model.predict_proba(X_test)[:, 1]

    # Stage 5: SHAP
    print("\n[5/6] Computing SHAP values...")
    from src.explain import compute_shap_explanations
    compute_shap_explanations(lgbm, X_test)

    # Stage 6: Threshold optimisation
    print("\n[6/6] Optimising decision threshold...")
    from src.threshold import expected_loss, optimize_threshold, sensitivity_table
    from src.config import SENSITIVITY_FILE

    # OOF predictions on training data for threshold selection
    oof_probs = np.zeros(len(y_train))
    for tr_idx, val_idx in cv.split(X_train, y_train):
        X_ft, X_fv = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_ft = y_train.iloc[tr_idx]
        fm = train_lgbm(X_ft, y_ft)
        oof_probs[val_idx] = fm.predict_proba(X_fv)[:, 1]

    opt_threshold, _ = optimize_threshold(y_train.values, oof_probs)
    print(f"  Optimal threshold (OOF): {opt_threshold:.4f}")

    loss_at_05 = expected_loss(y_test.values, y_prob_cal, 0.5)
    loss_at_opt = expected_loss(y_test.values, y_prob_cal, opt_threshold)
    loss_reduction = (1 - loss_at_opt / loss_at_05) * 100 if loss_at_05 > 0 else 0.0

    # Decile lift
    n_test = len(y_test)
    top_n = n_test // 10
    sorted_idx = np.argsort(y_prob_cal)[::-1]
    top_decile_idx = sorted_idx[:top_n]
    top_decile_defaults = y_test.values[top_decile_idx].sum()
    total_defaults = y_test.values.sum()
    top_decile_capture = float(top_decile_defaults / total_defaults)
    top_decile_default_rate = float(top_decile_defaults / top_n)
    overall_default_rate = float(y_test.values.mean())
    top_decile_lift = top_decile_default_rate / overall_default_rate

    # Sensitivity table
    sens_df = sensitivity_table(y_test.values, y_prob_cal)
    SENSITIVITY_FILE.parent.mkdir(parents=True, exist_ok=True)
    sens_df.to_csv(SENSITIVITY_FILE, index=False)

    # Threshold plot
    thresholds = np.arange(0.01, 1.0, 0.01)
    losses = [expected_loss(y_test.values, y_prob_cal, t) for t in thresholds]
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

    save_metrics("business", {
        "optimal_threshold": round(opt_threshold, 4),
        "loss_at_0.5": round(loss_at_05, 6),
        "loss_at_optimal": round(loss_at_opt, 6),
        "loss_reduction_pct": round(loss_reduction, 2),
        "top_decile_capture": round(top_decile_capture, 6),
        "top_decile_lift": round(top_decile_lift, 4),
    })

    # Update README
    print("\nUpdating README...")
    from src.report import update_readme
    update_readme()

    dt = time.time() - t0
    print(f"\nPipeline completed in {dt:.1f}s.")


if __name__ == "__main__":
    main()
