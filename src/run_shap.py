"""Run SHAP explainability."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import download_data, load_clean, split_data
from src.features import build_features
from src.models import train_lgbm
from src.explain import compute_shap_explanations


def main() -> None:
    """Compute SHAP explanations for LightGBM model."""
    download_data()
    df = load_clean()
    df = build_features(df)
    X_train, X_test, y_train, y_test = split_data(df)
    model = train_lgbm(X_train, y_train)
    compute_shap_explanations(model, X_test)


if __name__ == "__main__":
    main()
