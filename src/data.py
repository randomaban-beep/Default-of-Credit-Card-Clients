"""Data loading, cleaning and splitting utilities."""
from __future__ import annotations

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.config import DATA_RAW, SEED, TEST_SIZE

RAW_FILE: Path = DATA_RAW / "credit_default.csv"


def download_data() -> None:
    """Download the UCI credit-default dataset and save it locally.

    Uses ucimlrepo (dataset id 350). Falls back to OpenML if that fails.
    Skips the download if the file already exists.
    """
    if RAW_FILE.exists():
        print(f"Data already exists at {RAW_FILE}, skipping download.")
        return

    RAW_FILE.parent.mkdir(parents=True, exist_ok=True)

    try:
        from ucimlrepo import fetch_ucirepo  # type: ignore
        dataset = fetch_ucirepo(id=350)
        df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
        df.to_csv(RAW_FILE, index=False)
        print(f"Downloaded via ucimlrepo -> {RAW_FILE}")
        return
    except Exception as exc:
        print(f"ucimlrepo failed ({exc}), trying OpenML …")

    try:
        from sklearn.datasets import fetch_openml  # type: ignore
        bunch = fetch_openml(data_id=42477, as_frame=True, parser="auto")
        df = bunch.frame
        df.to_csv(RAW_FILE, index=False)
        print(f"Downloaded via OpenML -> {RAW_FILE}")
        return
    except Exception as exc2:
        raise RuntimeError(
            f"Both ucimlrepo and OpenML failed. Last error: {exc2}"
        ) from exc2


def load_clean() -> pd.DataFrame:
    """Load and clean the raw dataset.

    Renames PAY_0->PAY_1, remaps EDUCATION/MARRIAGE codes,
    renames target to 'default', drops ID.
    """
    df = pd.read_csv(RAW_FILE)

    # Normalise column names (ucimlrepo and OpenML differ in casing)
    df.columns = [c.upper().strip() for c in df.columns]

    # Map OpenML x1-x23 naming if present
    if "X1" in df.columns:
        col_map = {
            "X1": "LIMIT_BAL",
            "X2": "SEX",
            "X3": "EDUCATION",
            "X4": "MARRIAGE",
            "X5": "AGE",
            "X6": "PAY_1",
            "X7": "PAY_2",
            "X8": "PAY_3",
            "X9": "PAY_4",
            "X10": "PAY_5",
            "X11": "PAY_6",
            "X12": "BILL_AMT1",
            "X13": "BILL_AMT2",
            "X14": "BILL_AMT3",
            "X15": "BILL_AMT4",
            "X16": "BILL_AMT5",
            "X17": "BILL_AMT6",
            "X18": "PAY_AMT1",
            "X19": "PAY_AMT2",
            "X20": "PAY_AMT3",
            "X21": "PAY_AMT4",
            "X22": "PAY_AMT5",
            "X23": "PAY_AMT6",
            "Y": "default",
        }
        df = df.rename(columns=col_map)

    # Drop ID if present
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    # Rename PAY_0 -> PAY_1 (original dataset quirk)
    if "PAY_0" in df.columns:
        df = df.rename(columns={"PAY_0": "PAY_1"})

    # Rename target
    target_candidates = ["DEFAULT.PAYMENT.NEXT.MONTH", "DEFAULT_PAYMENT_NEXT_MONTH",
                         "Y", "DEFAULT"]
    for t in target_candidates:
        if t in df.columns:
            df = df.rename(columns={t: "default"})
            break

    # Remap EDUCATION: 0, 5, 6 -> 4 (unknown)
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})

    # Remap MARRIAGE: 0 -> 3 (unknown)
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

    # Cast target to int
    df["default"] = df["default"].astype(int)

    assert df.isnull().sum().sum() == 0, "Missing values found after cleaning!"
    return df


def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified 80/20 train/test split.

    Returns X_train, X_test, y_train, y_test.
    Split is performed BEFORE any fitting step.
    """
    X = df.drop(columns=["default"])
    y = df["default"]
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y,
    )


if __name__ == "__main__":
    download_data()
    df = load_clean()
    print(f"Shape: {df.shape}")
    print(f"Default rate: {df['default'].mean():.4f}")
    print(df.dtypes)
