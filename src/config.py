"""Central configuration for the project."""
from pathlib import Path

# Reproducibility
SEED: int = 42

# Paths
ROOT: Path = Path(__file__).parent.parent
DATA_RAW: Path = ROOT / "data" / "raw"
DATA_PROCESSED: Path = ROOT / "data" / "processed"
REPORTS: Path = ROOT / "reports"
FIGURES: Path = REPORTS / "figures"
METRICS_FILE: Path = REPORTS / "metrics.json"
SENSITIVITY_FILE: Path = REPORTS / "sensitivity.csv"

# Modelling
TEST_SIZE: float = 0.2

# Business costs
COST_FN: float = 5.0  # cost of approving a defaulter (false negative)
COST_FP: float = 1.0  # cost of declining a good customer (false positive)
