# Shared paths and settings

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

TICKERS = ["SPY", "QQQ", "IWM"]
START = "1993-01-01"  # SPY launch, QQQ and IWM start later

# CRSP PERMNOs, checked against crsp.stocknames in download_wrds.py
PERMNOS = {"SPY": 84398, "QQQ": 86755, "IWM": 88222}

# Train is for estimation, validation for choices, test stays untouched until the end
SAMPLES = {
    "train": ("1993-01-01", "2012-12-31"),
    "validation": ("2013-01-01", "2018-12-31"),
    "test": ("2019-01-01", "2025-12-31"),
}

# Last day covered by every raw source
END = "2025-12-31"

# Signal picked on validation in step 7, frozen before the test run
CHOSEN_SIGNAL = "sig_c"
