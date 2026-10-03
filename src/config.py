"""Shared settings for the download scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

TICKERS = ["SPY", "QQQ", "IWM"]
START = "1993-01-01"  # SPY launch; QQQ (1999) and IWM (2000) start later

# CRSP PERMNOs, checked against crsp.stocknames in download_wrds.py
PERMNOS = {"SPY": 84398, "QQQ": 86755, "IWM": 88222}

# Sample split. Statistics are estimated on train only; validation is for choices
# made along the way; test is not looked at until the end.
SAMPLES = {
    "train": ("1993-01-01", "2012-12-31"),
    "validation": ("2013-01-01", "2018-12-31"),
    "test": ("2019-01-01", "2099-12-31"),
}

# Last day every raw source covers (FRED ends 2026-10-02 in the current snapshot)
END = "2026-10-02"

# Step 7: signal chosen on validation (2013-2018) by the rule fixed beforehand:
# highest Sharpe at 1 bp per trade averaged over SPY, QQQ and IWM, ties (2 dp)
# broken by fewer trades. Frozen: do not change after looking at the test sample.
CHOSEN_SIGNAL = "sig_a"
