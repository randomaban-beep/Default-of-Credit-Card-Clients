"""Feature engineering — no fitting, pure transforms."""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to *df* (no fitting required).

    Features added
    --------------
    avg_bill_amt        : mean of BILL_AMT1-6
    avg_pay_amt         : mean of PAY_AMT1-6
    pay_to_bill_ratio   : avg_pay_amt / (avg_bill_amt + 1) to avoid div-by-zero
    months_delayed      : count of PAY_x columns where value > 0
    credit_utilization  : avg_bill_amt / (LIMIT_BAL + 1)
    max_delay           : maximum value across PAY_1-6
    """
    df = df.copy()

    bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
    pay_amt_cols = [f"PAY_AMT{i}" for i in range(1, 7)]
    pay_status_cols = ["PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

    df["avg_bill_amt"] = df[bill_cols].mean(axis=1)
    df["avg_pay_amt"] = df[pay_amt_cols].mean(axis=1)
    df["pay_to_bill_ratio"] = (df["avg_pay_amt"] / (df["avg_bill_amt"] + 1.0)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["months_delayed"] = (df[pay_status_cols] > 0).sum(axis=1)
    df["credit_utilization"] = (df["avg_bill_amt"] / (df["LIMIT_BAL"] + 1.0)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["max_delay"] = df[pay_status_cols].max(axis=1)

    return df
