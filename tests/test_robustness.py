import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robustness import VOL_TARGET_WINDOW, VOL_WINDOW, vol_weights

class VolWeightsTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.returns = pd.DataFrame(rng.normal(0, 0.01, (400, 3)),
                                    index=pd.bdate_range("2005-01-03", periods=400),
                                    columns=["overnight", "intraday", "ret"])

    def test_buy_hold_ignores_entry_close_and_later_returns(self):
        r = self.returns
        entry, next_open = r.index[330], r.index[331]
        before = vol_weights(r)
        changed = r.copy()
        changed.loc[entry:, "ret"] = 0.5
        after = vol_weights(changed)

        # The weight for the next overnight leg is set before the entry day's close
        pd.testing.assert_series_equal(before.loc[:next_open, "ret"], after.loc[:next_open, "ret"])
        self.assertNotEqual(before.loc[r.index[332], "ret"], after.loc[r.index[332], "ret"])

    def test_other_legs_keep_their_existing_lag(self):
        r = self.returns[["overnight", "intraday"]]
        vol = r.rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std().shift(1)
        expected = (vol.rolling(VOL_TARGET_WINDOW, min_periods=VOL_TARGET_WINDOW).median() / vol).clip(upper=1.0)
        pd.testing.assert_frame_equal(vol_weights(self.returns)[r.columns], expected)

if __name__ == "__main__":
    unittest.main()
