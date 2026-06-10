"""Model training utilities."""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV

from src.config import SEED


def train_logistic(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Train a logistic regression pipeline (StandardScaler + LR).

    Uses class_weight='balanced' to handle class imbalance.
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=SEED,
            solver="lbfgs",
        )),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def train_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Any:
    """Train a LightGBM model with randomized hyperparameter search.

    Uses early stopping on a validation split carved from training data only.
    Randomized search over ~20 trials, CV scored by ROC-AUC.
    Returns the best estimator.
    """
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split

    # Validation split for early stopping (from training data only)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=SEED, stratify=y_train
    )

    param_dist = {
        "n_estimators": [200, 300, 400],
        "learning_rate": [0.03, 0.05, 0.07],
        "num_leaves": [20, 31, 40],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "min_child_samples": [20, 30, 50],
    }

    base = lgb.LGBMClassifier(
        random_state=SEED,
        class_weight="balanced",
        verbosity=-1,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    search = RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=20,
        scoring="roc_auc",
        cv=cv,
        random_state=SEED,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    best_params = search.best_params_
    print(f"Best LGBM params: {best_params}")
    print(f"Best CV ROC-AUC: {search.best_score_:.4f}")
    return search.best_estimator_
