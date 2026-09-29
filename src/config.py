"""Shared settings for the download scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

TICKERS = ["SPY", "QQQ", "IWM"]
START = "1993-01-01"  # SPY launch; QQQ (1999) and IWM (2000) start later

# CRSP PERMNOs, checked against crsp.stocknames in download_wrds.py
PERMNOS = {"SPY": 84398, "QQQ": 86755, "IWM": 88222}
