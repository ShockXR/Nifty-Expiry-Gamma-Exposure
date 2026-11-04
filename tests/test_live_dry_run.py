import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import live_dry_run as live  # noqa: E402


def _bars(day="2026-11-03", seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range(f"{day} 09:15", f"{day} 15:29", freq="1min")
    close = 23000 * np.exp(np.cumsum(rng.normal(0, 3e-4, len(ts))))
    return pd.DataFrame({"timestamp": ts, "open": close, "high": close, "low": close, "close": close}), close


def test_minute_rv_splits_add_up_and_use_session_slots():
    b, close = _bars()
    rv = live.minute_rv(b).iloc[0]
    r = np.diff(np.log(close))
    assert np.isclose(rv["rv_day"], (r ** 2).sum())
    assert np.isclose(rv["rv_to_1430"] + rv["rv_last_hour"], rv["rv_day"])
    assert np.isclose(rv["rv_last_hour"], (r[-60:] ** 2).sum())  # returns ending 14:30 to 15:29
    assert np.isclose(rv["p_1459"], close[344]) and np.isclose(rv["p_1529"], close[374])


def test_apply_1430_uses_frozen_coefficients():
    rule = {"coefficients": {"const": 0.5, "log_rv_to_1430": 0.8, "lvix_prev": 0.1},
            "predetermined_inputs": {"lvix_prev": 2.0}, "smearing": 1.0, "resid_q10": -0.5, "resid_q90": 0.5}
    out = live.apply_1430(rule, 1e-5)
    assert np.isclose(np.log(out["forecast"]), 0.5 + 0.8 * np.log(1e-5) + 0.2)
    assert out["p10"] < out["forecast"] < out["p90"]
