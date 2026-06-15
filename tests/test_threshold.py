"""Tests for threshold.py."""
from __future__ import annotations

import numpy as np
import pytest
from src.threshold import expected_loss, optimize_threshold


def test_high_cost_fn_gives_low_threshold() -> None:
    """When FN cost >> FP cost, optimal threshold should be below 0.5."""
    rng = np.random.default_rng(42)
    n = 2000
    score = rng.normal(0, 1, n)
    prob = 1 / (1 + np.exp(-score))
    y_true = (rng.uniform(0, 1, n) < prob).astype(int)
    t, _ = optimize_threshold(y_true, prob, cost_fn=10, cost_fp=1)
    assert t < 0.5, f"Expected threshold < 0.5, got {t}"


def test_equal_costs_near_half() -> None:
    """With equal costs and calibrated probs, optimal threshold should be near 0.5."""
    rng = np.random.default_rng(42)
    n = 5000
    y_true = rng.integers(0, 2, n)
    # Perfectly calibrated: prob == true label rate
    y_prob = np.where(y_true == 1, rng.uniform(0.45, 0.75, n), rng.uniform(0.25, 0.55, n))
    t, _ = optimize_threshold(y_true, y_prob, cost_fn=1, cost_fp=1)
    assert 0.3 <= t <= 0.7, f"Expected threshold near 0.5, got {t}"


def test_expected_loss_all_approve() -> None:
    """Threshold of 1.0 approves everyone: cost = FN rate * cost_fn."""
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.3, 0.4, 0.6, 0.7])
    loss = expected_loss(y_true, y_prob, threshold=1.0, cost_fn=5, cost_fp=1)
    # All defaulters approved => 2 FN, 0 FP => cost = 2*5/4 = 2.5
    assert loss == pytest.approx(2.5, abs=1e-6)
