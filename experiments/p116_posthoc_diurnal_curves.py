"""P116, POST HOC and descriptive (added after the registered run; not in the plan): the diurnal curves of the last
Nifty refit (training days 0 to 926), raw and with the 5-minute smoothing, one curve and the expiry-day curve, at the
open, around 15:00 and at the close. It explains where the smoothing and the expiry-day curve change mcsGARCH's
forecasts (docs/P116_BENCHMARK_AS_PRINTED.md, Section 4). No forecast or loss is computed. REAL DATA.

    PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p116_posthoc_diurnal_curves.py \\
        > results/p116_benchmark_as_printed/posthoc_diurnal_curves.txt
"""

import argparse
import importlib.util
from pathlib import Path

import pandas as pd

from niftygex.volatility import McsGarch, daily_egarch_variance

REPO = Path(__file__).resolve().parents[1]


def main():
    spec = importlib.util.spec_from_file_location("p116", REPO / "experiments" / "p116_benchmark_as_printed.py")
    p116 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p116)
    ex = p116.ex
    st = ex.gd.PIPELINE_SETTINGS
    ns = argparse.Namespace(source="nse", raw=str(REPO / "data" / "raw"), symbol="NIFTY", oi_unit=st["oi_unit"],
                            lot_size=None, rate=st["rate"], start=st["start"], end=ex.INDEXES["NIFTY"]["end"],
                            min_coverage=st["min_coverage"])
    m = ex.rp.load_market(ns)
    R = m["R"]
    is_exp = m["labels"]["is_expiry"].to_numpy()
    T, M = R.shape
    start = 150 + 21 * 37          # the last refit
    h, _ = daily_egarch_variance(R.sum(axis=1), train_size=start)
    out = {}
    for name, kw in (("raw_one_curve", dict(expiry_diurnal=False, smooth=1)),
                     ("smoothed_one_curve", dict(expiry_diurnal=False, smooth=5)),
                     ("raw_expiry_curve", dict(expiry_diurnal=True, smooth=1)),
                     ("smoothed_expiry_curve", dict(expiry_diurnal=True, smooth=5))):
        g = McsGarch(**kw).fit(R[:start], h[:start], is_exp[:start])
        out[name] = g.s if "one" in name else g.s_expiry
    clock = [(pd.Timestamp("2000-01-01 09:15") + pd.Timedelta(minutes=i)).strftime("%H:%M") for i in range(M)]
    df = pd.DataFrame(out, index=clock)
    mins = list(range(0, 6)) + list(range(340, 351)) + list(range(370, 375))
    pd.set_option("display.width", 200)
    print("REAL DATA, Nifty, last refit (training days 0 to", start - 1, "), diurnal factors s_i:")
    print(df.iloc[mins].round(3).to_string())
    for name in out:
        s = out[name]
        print(name, "sum 14:45-14:59", round(float(s[330:345].sum()), 3), "sum 15:00-15:14",
              round(float(s[345:360].sum()), 3), "sum 09:15-09:29", round(float(s[0:15].sum()), 3))
    print("expiry days in training:", int(is_exp[:start].sum()))


if __name__ == "__main__":
    main()
