"""Tests for the real-data fixes made when the loaders first met live NSE files."""

import numpy as np
import pandas as pd

from niftygex.data import actual_expiries, build_market_from_nse, parity_forward
from niftygex.expiry_calendar import label_days
from niftygex.simulate import SimConfig, export_nse_layout, simulate_market


def test_actual_expiries_picks_holiday_shift():
    # Contract nominally for Thursday 26 Jan 2023 (holiday) really expires Wednesday 25 Jan.
    def tbl(d, exps):
        return pd.DataFrame({"date": pd.Timestamp(d), "expiry": pd.to_datetime(exps)})
    bhav = {pd.Timestamp(d): tbl(d, e) for d, e in [
        ("2023-01-24", ["2023-01-25", "2023-02-02"]),
        ("2023-01-25", ["2023-01-25", "2023-02-02"]),
        ("2023-01-27", ["2023-02-02"]),
    ]}
    assert list(actual_expiries(bhav)) == [pd.Timestamp("2023-01-25")]
    lab = label_days(pd.to_datetime(["2023-01-24", "2023-01-25", "2023-01-27"]), expiries=actual_expiries(bhav))
    assert lab["is_expiry"].tolist() == [False, True, False]


def test_parity_forward_recovers_forward():
    from niftygex.greeks import price as black76_price
    F, T, r = 22150.0, 3 / 252, 0.065
    K = np.arange(21500, 22801, 50.0)
    rows = []
    for k in K:
        for is_call in (True, False):
            rows.append({"strike": k, "is_call": is_call, "settle": float(black76_price(F, k, T, r, 0.13, is_call))})
    assert abs(parity_forward(pd.DataFrame(rows), T, r) - F) < 1e-6


def test_vix_iso_dates_not_read_day_first(tmp_path):
    m = simulate_market(SimConfig(n_days=12, kappa=0.0, seed=4))
    paths = export_nse_layout(m, tmp_path)
    v = pd.read_csv(paths["vix"])
    v["date"] = pd.to_datetime(v["date"], dayfirst=True).dt.strftime("%Y-%m-%d")  # ISO, as download_nse.py writes
    v.to_csv(paths["vix"], index=False)
    mk = build_market_from_nse(paths["bars"], paths["bhavcopy"], paths["participant"], paths["vix"], oi_unit="keep_units")
    assert np.allclose(mk["vix"], m["vix"], rtol=1e-6)
    assert mk["lot_size"] == 1.0
    # keep_units leaves OI in shares: contracts times the simulator's lot size
    key = ["is_call", "strike"]
    got = mk["chains"][3].sort_values(key)["oi"].to_numpy()
    want = m["chains"][3].sort_values(key)["oi"].to_numpy() * m["lot_size"]
    assert np.allclose(got, want)


def test_load_bhavcopy_sorts_strikes(tmp_path):
    from niftygex.data import load_bhavcopy
    rows = []
    for k, typ in [(22500, "CE"), (21000, "CE"), (22000, "PE"), (23000, "CE"), (21500, "PE")]:
        rows.append({"TradDt": "2026-10-26", "FinInstrmTp": "IDO", "TckrSymb": "NIFTY", "XpryDt": "2026-10-27",
                     "StrkPric": k, "OptnTp": typ, "ClsPric": 1.0, "SttlmPric": 1.0, "UndrlygPric": 22750.0,
                     "OpnIntrst": 650, "ChngInOpnIntrst": 0, "TtlTradgVol": 1, "NewBrdLotQty": 65})
    f = tmp_path / "fo_20261026.csv"
    pd.DataFrame(rows).to_csv(f, index=False)
    b = load_bhavcopy(f)
    for is_call in (True, False):
        k = b.loc[b["is_call"] == is_call, "strike"].to_numpy()
        assert np.all(np.diff(k) > 0)


def test_holiday_adjusted_expiries_maps_to_previous_trade_date():
    from niftygex.data import holiday_adjusted_expiries
    def chain(d, exps):
        return pd.DataFrame({"date": pd.Timestamp(d), "expiry": pd.to_datetime(exps)})
    bhav = {
        pd.Timestamp("2023-06-26"): chain("2023-06-26", ["2023-06-29", "2023-07-06"]),
        pd.Timestamp("2023-06-27"): chain("2023-06-27", ["2023-06-29", "2023-07-06"]),
        pd.Timestamp("2023-06-28"): chain("2023-06-28", ["2023-06-28", "2023-07-06"]),
        pd.Timestamp("2023-06-30"): chain("2023-06-30", ["2023-07-06", "2023-07-13"]),
    }
    m = holiday_adjusted_expiries(bhav)
    assert m == {pd.Timestamp("2023-06-29"): pd.Timestamp("2023-06-28")}  # 6 and 13 July are after the last file
