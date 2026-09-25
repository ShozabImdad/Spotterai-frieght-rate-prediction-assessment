from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"

TRAIN_PATH = DATA_DIR / "train_test.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
TEMPLATE_PATH = DATA_DIR / "validation_predictions_template.csv"
DECEMBER_PATH = DATA_DIR / "december_chart_inputs.csv"
PREDICTIONS_PATH = ROOT / "validation_predictions.csv"

TARGET = "posted_rate"
SEED = 42

# Labels whose out-of-fold log residual exceeds this are treated as corrupted.
# Clean residuals sit within roughly +/-0.12, corrupted ones beyond +/-1.0.
LABEL_OUTLIER_THRESHOLD = 0.3

BACKTEST_MONTHS = ["2025-08", "2025-09", "2025-10"]
