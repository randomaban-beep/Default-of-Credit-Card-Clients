"""Cost-based threshold optimisation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import COST_FN, COST_FP


def expected_loss(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    cost_fn: float = COST_FN,
    cost_fp: float = COST_FP,
) -> float:
    """Average cost per account given a decision threshold.

    Approving someone who defaults = FN (missed detection) costs cost_fn.
    Declining a good customer = FP costs cost_fp.
    """
    y_pred = (y_prob >= threshold).astype(int)  # 1 = decline (predict default)
    fn = ((y_true == 1) & (y_pred == 0)).sum()  # defaulters approved (not caught)
    fp = ((y_true == 0) & (y_pred == 1)).sum()  # good customers declined
    total_cost = fn * cost_fn + fp * cost_fp
    return float(total_cost / len(y_true))


def optimize_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fn: float = COST_FN,
    cost_fp: float = COST_FP,
) -> tuple[float, float]:
    """Scan thresholds 0.01 to 0.99 and return (best_threshold, min_loss)."""
    thresholds = np.arange(0.01, 1.0, 0.01)
    losses = [expected_loss(y_true, y_prob, t, cost_fn, cost_fp) for t in thresholds]
    best_idx = int(np.argmin(losses))
    return float(thresholds[best_idx]), float(losses[best_idx])


def sensitivity_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> pd.DataFrame:
    """Optimal threshold and losses for different cost ratios."""
    ratios = [(2, 1), (3, 1), (5, 1), (10, 1)]
    rows = []
    for fn_cost, fp_cost in ratios:
        t, loss = optimize_threshold(y_true, y_prob, fn_cost, fp_cost)
        loss_05 = expected_loss(y_true, y_prob, 0.5, fn_cost, fp_cost)
        rows.append({
            "cost_fn": fn_cost,
            "cost_fp": fp_cost,
            "cost_ratio": f"{fn_cost}:{fp_cost}",
            "optimal_threshold": round(t, 4),
            "loss_at_optimal": round(loss, 6),
            "loss_at_0.5": round(loss_05, 6),
            "loss_reduction_pct": round((1 - loss / loss_05) * 100, 2) if loss_05 > 0 else 0.0,
        })
    return pd.DataFrame(rows)
