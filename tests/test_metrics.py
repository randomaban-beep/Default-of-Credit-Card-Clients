"""Tests for evaluate.py metrics functions."""
from __future__ import annotations

import numpy as np
import pytest
from src.evaluate import compute_metrics, ks_statistic, gini


def test_perfect_predictions_auc() -> None:
    """Perfect classifier should yield ROC-AUC = 1."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.0, 0.1, 0.2, 0.8, 0.9, 1.0])
    metrics = compute_metrics(y_true, y_prob)
    assert metrics["roc_auc"] == pytest.approx(1.0, abs=1e-6)


def test_perfect_predictions_ks() -> None:
    """Perfect classifier should yield KS = 1."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.0, 0.1, 0.2, 0.8, 0.9, 1.0])
    assert ks_statistic(y_true, y_prob) == pytest.approx(1.0, abs=1e-6)


def test_random_predictions_auc() -> None:
    """Random predictions should yield ROC-AUC near 0.5."""
    rng = np.random.default_rng(42)
    n = 10_000
    y_true = rng.integers(0, 2, n)
    y_prob = rng.uniform(0, 1, n)
    metrics = compute_metrics(y_true, y_prob)
    assert 0.45 < metrics["roc_auc"] < 0.55


def test_gini_equals_2auc_minus_1() -> None:
    """Gini coefficient must equal 2*AUC - 1."""
    rng = np.random.default_rng(7)
    y_true = rng.integers(0, 2, 1000)
    y_prob = rng.uniform(0, 1, 1000)
    metrics = compute_metrics(y_true, y_prob)
    expected_gini = 2 * metrics["roc_auc"] - 1
    assert metrics["gini"] == pytest.approx(expected_gini, abs=1e-6)
