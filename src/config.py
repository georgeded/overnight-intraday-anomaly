from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

TICKERS = ["SPY", "QQQ", "IWM"]
START = "1993-01-01"
END = "2025-12-31"
PERMNOS = {"SPY": 84398, "QQQ": 86755, "IWM": 88222}
SAMPLES = {"train": ("1993-01-01", "2012-12-31"), "validation": ("2013-01-01", "2018-12-31"), "test": ("2019-01-01", "2025-12-31")}
CHOSEN_SIGNAL = "sig_c"  # picked on validation in step 7

def md_table(df, floatfmt="{:.4f}"):
    if df.index.name is not None or isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(lines)

def write_report(name, lines):
    path = ROOT / "reports" / name
    path.write_text("\n".join(lines))
    print(f"wrote {path.relative_to(ROOT)}")
