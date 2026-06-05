"""Project configuration: paths, constants, RNG seed."""
from pathlib import Path

ROOT = Path(__file__).absolute().parent.parent

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

for d in (DATA_RAW, DATA_PROCESSED, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Classification target bins on G3 (0-20 scale)
CLASS_BINS = [-0.1, 9.5, 14.5, 20.1]
CLASS_LABELS = ["At Risk", "Average", "Excellent"]

# Drop G1/G2 by default to avoid label leakage (they're prior-period grades)
DROP_LEAKAGE_FEATURES = True
LEAKAGE_COLS = ["G1", "G2"]

TARGET_REG = "G3"
TARGET_CLF = "performance_class"
