"""The --symbol option of scripts/run_pipeline.py (pre-registration P91, docs/prereg/P91_BANKNIFTY_FORECAST.md).

Two promises are tested:

1. The Nifty path is unchanged. Synthetic files with both indices give the same Nifty market as files
   with Nifty rows only; the pipeline's wiring for NIFTY is the old one (Nifty bars, "Nifty 50" official
   days, column names); and, on real data, the rebuilt Nifty daily panels equal the committed
   results/pipeline_nse_v2 files.
2. The Bank Nifty path reads Bank Nifty's own bars, chains, lot sizes, strike spacing, futures and
   expiry calendar, and the "other index expires today" feature then refers to Nifty.

The real-data tests look only at calendar facts, option chains and price LEVELS (checked against NSE's
official close). They compute no Bank Nifty return-based quantity: phase 1 of P91 forbids looking at any
Bank Nifty outcome before the pre-registration is committed.
"""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from niftygex.data import build_market_from_nse, load_bhavcopy_all
from niftygex.features import build_dataset, build_gex_panel
from niftygex.greeks import price as black76_price

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_pipeline as rp  # noqa: E402

RAW = ROOT / "data" / "raw"
V2 = ROOT / "results" / "pipeline_nse_v2"


# ---------------------------------------------------------------------------
# Synthetic NSE-layout files holding BOTH indices (SYNTHETIC data, made up for the test)
# ---------------------------------------------------------------------------

DAYS = pd.bdate_range("2024-01-01", "2024-02-29")  # Monday 1 Jan to Thursday 29 Feb
SPEC = {
    # spot level, strike step, lot size, expiry weekday (Thu = 3, Wed = 2), futures traded value per day
    "NIFTY": dict(spot=22000.0, step=50.0, lot=25, wd=3, fut_value=1.0e9),
    "BANKNIFTY": dict(spot=48000.0, step=100.0, lot=15, wd=2, fut_value=2.0e9),
}


def _weekly(wd):
    cal = pd.date_range("2024-01-01", "2024-04-30")
    return pd.DatetimeIndex([d for d in cal if d.weekday() == wd])


def _monthly(wd):
    w = _weekly(wd)
    return pd.DatetimeIndex([g.max() for _, g in pd.Series(w, index=w).groupby([w.year, w.month])])


def _rows(sym, d, t):
    s = SPEC[sym]
    spot = s["spot"] * (1 + 0.001 * t)
    weekly = _weekly(s["wd"])
    exps = weekly[weekly >= d][:2]  # the expiring contract trades on its expiry day, as in real files
    out = []
    strikes = np.round(spot / s["step"]) * s["step"] + s["step"] * np.arange(-10, 11)
    for e in exps:
        T = max(int(np.busday_count(d.date(), e.date())), 1) / 252
        for j, k in enumerate(strikes):
            for typ in ("CE", "PE"):
                out.append({"TradDt": d.strftime("%Y-%m-%d"), "FinInstrmTp": "IDO", "TckrSymb": sym,
                            "XpryDt": e.strftime("%Y-%m-%d"), "StrkPric": k, "OptnTp": typ,
                            "ClsPric": float(black76_price(spot, k, T, 0.065, 0.15, typ == "CE")),
                            "SttlmPric": float(black76_price(spot, k, T, 0.065, 0.15, typ == "CE")),
                            "UndrlygPric": spot, "OpnIntrst": s["lot"] * (100 + 7 * j + (typ == "PE")),
                            "ChngInOpnIntrst": 0, "TtlTradgVol": 10, "NewBrdLotQty": s["lot"], "TtlTrfVal": 0.0})
    fe = _monthly(s["wd"])
    fe = fe[fe >= d][0]
    out.append({"TradDt": d.strftime("%Y-%m-%d"), "FinInstrmTp": "IDF", "TckrSymb": sym, "XpryDt": fe.strftime("%Y-%m-%d"),
                "StrkPric": np.nan, "OptnTp": "", "ClsPric": spot, "SttlmPric": spot, "UndrlygPric": spot,
                "OpnIntrst": 1000.0, "ChngInOpnIntrst": 0, "TtlTradgVol": 100, "NewBrdLotQty": s["lot"],
                "TtlTrfVal": s["fut_value"]})
    return out


def _bars(sym):
    s = SPEC[sym]
    rows = []
    for t, d in enumerate(DAYS):
        ts = d + pd.Timedelta(hours=9, minutes=15) + pd.to_timedelta(np.arange(375), unit="min")
        level = s["spot"] * (1 + 0.001 * t) * (1 + 0.0005 * np.sin(np.arange(375) / 7 + t))
        rows.append(pd.DataFrame({"timestamp": ts, "open": level, "high": level * 1.0002, "low": level * 0.9998,
                                  "close": level}))
    return pd.concat(rows)


@pytest.fixture(scope="module")
def two_index_files(tmp_path_factory):
    root = tmp_path_factory.mktemp("two_index")
    both, nifty_only = root / "bhav_both", root / "bhav_nifty_only"
    both.mkdir()
    nifty_only.mkdir()
    for t, d in enumerate(DAYS):
        n, b = pd.DataFrame(_rows("NIFTY", d, t)), pd.DataFrame(_rows("BANKNIFTY", d, t))
        name = f"BhavCopy_NSE_FO_0_0_0_{d:%Y%m%d}_F_0000.csv"
        pd.concat([b, n]).to_csv(both / name, index=False)  # Bank Nifty rows first, to test the filter
        n.to_csv(nifty_only / name, index=False)
    _bars("NIFTY").to_csv(root / "nifty_1min.csv", index=False)
    _bars("BANKNIFTY").to_csv(root / "banknifty_1min.csv", index=False)
    return root


def _build(root, sym, bhav="bhav_both", **kw):
    bars = root / ("nifty_1min.csv" if sym == "NIFTY" else "banknifty_1min.csv")
    kw.setdefault("oi_unit", "units")
    return build_market_from_nse(bars, root / bhav, symbol=sym, use_file_expiries=True, **kw)


def _in_days(idx):
    return [d for d in pd.DatetimeIndex(idx) if d in set(DAYS)]


def test_synthetic_banknifty_market_uses_bank_nifty_chain_calendar_and_lots(two_index_files):
    m = _build(two_index_files, "BANKNIFTY")
    lab = m["labels"]
    days = pd.DatetimeIndex(m["days"])
    assert list(days) == list(DAYS)
    # expiry calendar: Bank Nifty's Wednesdays, monthly = days on which a BANKNIFTY future expires
    assert list(days[lab["is_expiry"].to_numpy()]) == _in_days(_weekly(2))
    assert list(days[lab["is_monthly_expiry"].to_numpy()]) == _in_days(_monthly(2))
    # the "other index" is Nifty: its Thursdays
    assert _in_days(m["other_index_expiries"]) == _in_days(_weekly(3))
    for t, d in enumerate(days):
        ch = m["chains"][t]
        k = np.sort(ch["strike"].unique())
        assert set(np.diff(k)) == {100.0}                      # Bank Nifty strike spacing
        assert ch.attrs["expiry"] == _weekly(2)[_weekly(2) > d][0]  # nearest Bank Nifty expiry after d
        assert 0.9 < ch.attrs["spot"] / (48000.0 * (1 + 0.001 * t)) < 1.1
        # OI converted with Bank Nifty's lot size (15): contracts = 100 + 7 j (+1 for puts)
        assert np.all((ch["oi"].to_numpy() - 100) % 1 == 0) and ch["oi"].min() >= 100
    assert m["lot_size"] == 15.0
    assert np.allclose(m["futures"]["fut_value"].to_numpy(), 2.0e9)  # Bank Nifty futures, not Nifty's
    for d, tab in m["option_tables"].items():
        assert np.all(tab["strike"] % 100 == 0)
        assert set(tab["expiry"]) <= set(_weekly(2))
    # keep_units (the pipeline's choice): OI stays in shares and the lot size is 1
    mk = _build(two_index_files, "BANKNIFTY", oi_unit="keep_units")
    assert mk["lot_size"] == 1.0
    assert np.allclose(mk["chains"][5]["oi"].to_numpy(), m["chains"][5]["oi"].to_numpy() * 15)


def test_synthetic_banknifty_other_index_feature_is_nifty_expiry(two_index_files):
    m = _build(two_index_files, "BANKNIFTY", oi_unit="keep_units")
    ds = build_dataset(m, build_gex_panel(m), horizon=15)
    day = ds.groupby("date")["bn_expiry_only"].first()
    want = pd.Series([d.weekday() == 3 for d in day.index], index=day.index)  # Nifty Thursday, not a BN Wednesday
    assert day.astype(bool).equals(want)
    assert ds.loc[ds["is_expiry"], "date"].dt.weekday.eq(2).all()


def test_synthetic_nifty_market_ignores_bank_nifty_rows(two_index_files):
    m_both = _build(two_index_files, "NIFTY")
    m_only = _build(two_index_files, "NIFTY", bhav="bhav_nifty_only")
    pd.testing.assert_frame_equal(m_both["labels"], m_only["labels"])
    for a, b in zip(m_both["chains"], m_only["chains"]):
        pd.testing.assert_frame_equal(a, b)
        assert a.attrs == b.attrs
        assert set(np.diff(np.sort(a["strike"].unique()))) == {50.0}
    for d in m_both["option_tables"]:
        pd.testing.assert_frame_equal(m_both["option_tables"][d], m_only["option_tables"][d])
    pd.testing.assert_frame_equal(m_both["futures"], m_only["futures"])
    assert m_both["lot_size"] == 25.0
    assert list(m_both["days"][m_both["labels"]["is_expiry"].to_numpy()]) == _in_days(_weekly(3))
    # Nifty's "other index" is still Bank Nifty (the original design), and the default
    assert _in_days(m_both["other_index_expiries"]) == _in_days(_weekly(2))
    assert len(m_only["other_index_expiries"]) == 0
    m_default = build_market_from_nse(two_index_files / "nifty_1min.csv", two_index_files / "bhav_both",
                                      oi_unit="units", use_file_expiries=True)
    assert list(m_default["other_index_expiries"]) == list(m_both["other_index_expiries"])


# ---------------------------------------------------------------------------
# Pipeline wiring (no data needed)
# ---------------------------------------------------------------------------


def _capture_load_market(monkeypatch, tmp_path, **attrs):
    import niftygex.data as data_mod
    pd.DataFrame({"date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                  "index": ["Nifty 50", "Nifty Bank", "India VIX"]}).to_csv(tmp_path / "index_daily.csv", index=False)
    got = {}

    def fake(*args, **kw):
        got["args"], got["kw"] = args, kw
        return {}
    monkeypatch.setattr(data_mod, "build_market_from_nse", fake)
    a = types.SimpleNamespace(source="nse", raw=str(tmp_path), oi_unit="keep_units", lot_size=None, rate=0.065,
                              start="2024-01-01", end=None, min_coverage=0.95, **attrs)
    rp.load_market(a)
    return got


def test_load_market_wiring_nifty_is_the_old_one(monkeypatch, tmp_path):
    # no `symbol` attribute at all, as in scripts/forward_test.py, which predates --symbol
    got = _capture_load_market(monkeypatch, tmp_path)
    assert got["args"][0] == tmp_path / "nifty_1min.csv"
    assert got["args"][1] == tmp_path / "bhavcopy"
    assert got["kw"]["symbol"] == "NIFTY"
    assert list(pd.DatetimeIndex(got["kw"]["official_days"])) == [pd.Timestamp("2024-01-01")]
    assert "other_symbol" not in got["kw"]  # so the Nifty run keeps BANKNIFTY as its other index


def test_load_market_wiring_banknifty(monkeypatch, tmp_path):
    got = _capture_load_market(monkeypatch, tmp_path, symbol="BANKNIFTY")
    assert got["args"][0] == tmp_path / "banknifty_1min.csv"
    assert got["kw"]["symbol"] == "BANKNIFTY"
    assert list(pd.DatetimeIndex(got["kw"]["official_days"])) == [pd.Timestamp("2024-01-02")]
    assert got["kw"]["oi_unit"] == "keep_units" and got["kw"]["use_file_expiries"] is True


def test_daily_market_futures_column_name(tmp_path):
    days = pd.bdate_range("2024-01-01", periods=2)
    lab = pd.DataFrame({"is_expiry": False, "is_monthly_expiry": False, "regime": "multi_weekly", "weekday": 0,
                        "trading_days_to_expiry": 1.0}, index=days)
    ch = pd.DataFrame(columns=["strike", "is_call", "oi", "settle"])
    m = {"R": np.zeros((2, 375)), "close": np.ones((2, 375)), "open": np.ones((2, 375)), "labels": lab, "days": days,
         "vix": [0.1, 0.1], "chains": [ch, ch], "futures": pd.DataFrame({"fut_value": [1.0, 2.0]}, index=days)}
    rp.write_daily_market(m, tmp_path / "n.csv")
    rp.write_daily_market(m, tmp_path / "b.csv", symbol="BANKNIFTY")
    assert "nifty_futures_traded_value" in pd.read_csv(tmp_path / "n.csv").columns
    assert "banknifty_futures_traded_value" in pd.read_csv(tmp_path / "b.csv").columns


def test_symbol_rejected_for_synthetic_source():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_pipeline.py"), "--source", "synthetic",
                        "--symbol", "BANKNIFTY", "--days", "5"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 2 and "--symbol applies to --source nse only" in r.stderr


# ---------------------------------------------------------------------------
# Real data (skipped when data/raw is absent). REAL DATA, no Bank Nifty outcome is computed.
# ---------------------------------------------------------------------------

needs_raw = pytest.mark.skipif(not (RAW / "banknifty_1min.csv").exists() or not (RAW / "bhavcopy").exists()
                               or not (RAW / "index_daily.csv").exists(), reason="real NSE data not in data/raw")


def _real(symbol, start, end, **kw):
    a = types.SimpleNamespace(source="nse", raw=str(RAW), symbol=symbol, oi_unit="keep_units", lot_size=None,
                              rate=0.065, start=start, end=end, min_coverage=0.95)
    if kw:
        return build_market_from_nse(RAW / rp.SYMBOLS[symbol]["bars"], RAW / "bhavcopy", symbol=symbol,
                                     use_file_expiries=True, start=start, end=end, min_coverage=0.95, **kw)
    return rp.load_market(a)


@pytest.fixture(scope="module")
def bn_real():
    # 21 Oct 2024 to 7 Feb 2025: the last Bank Nifty weeklies, the switch to monthly only (20 Nov 2024),
    # a holiday-moved monthly (24 Dec 2024) and the first last-Thursday monthly (30 Jan 2025)
    return _real("BANKNIFTY", "2024-10-21", "2025-02-07")


@needs_raw
def test_real_banknifty_expiry_calendar(bn_real):
    m = bn_real
    days = pd.DatetimeIndex(m["days"])
    lab = m["labels"]
    D = pd.to_datetime
    assert list(days[lab["is_expiry"].to_numpy()]) == list(D(["2024-10-23", "2024-10-30", "2024-11-06", "2024-11-13",
                                                                "2024-11-27", "2024-12-24", "2025-01-30"]))
    assert list(days[lab["is_monthly_expiry"].to_numpy()]) == list(D(["2024-10-30", "2024-11-27", "2024-12-24",
                                                                        "2025-01-30"]))
    other = [d for d in pd.DatetimeIndex(m["other_index_expiries"]) if days[0] <= d <= days[-1]]
    # Nifty's Thursday weeklies are the "other index" expiries
    assert len(other) == 16 and all(d.weekday() == 3 for d in other)
    assert D("2024-11-14") in other and D("2024-11-21") in other
    # official "Nifty Bank" days only, and 1 Nov 2024 (Muhurat session: 60 evening bars, none in the 09:15 to
    # 15:30 session) is dropped by the 95% coverage rule
    idx = pd.read_csv(RAW / "index_daily.csv", parse_dates=["date"])
    official = set(idx.loc[idx["index"] == "Nifty Bank", "date"])
    assert set(days) <= official and D("2024-11-01") not in set(days)
    # regime labels: the shared regime dates also mark Bank Nifty's own rule changes
    assert lab.loc["2024-11-19", "regime"] == "multi_weekly" and lab.loc["2024-11-21", "regime"] == "single_weekly_thu"


@needs_raw
def test_real_banknifty_chains_bars_and_spacing(bn_real):
    m = bn_real
    days = pd.DatetimeIndex(m["days"])
    idx = pd.read_csv(RAW / "index_daily.csv", parse_dates=["date"])
    close = idx[idx["index"] == "Nifty Bank"].set_index("date")["close"]
    nifty_close = idx[idx["index"] == "Nifty 50"].set_index("date")["close"]
    assert not m["missing_bhavcopy_days"]
    for t, d in enumerate(days):
        ch = m["chains"][t]
        # Bank Nifty bars: the last bar is within 1% of NSE's official Bank Nifty close (a level, not a return),
        # and far from the Nifty 50 close
        assert abs(ch.attrs["spot_index"] / close[d] - 1) < 0.01
        assert ch.attrs["spot_index"] > 1.5 * nifty_close[d]
        assert abs(ch.attrs["spot"] / close[d] - 1) < 0.03  # parity forward of the Bank Nifty chain
        k = np.sort(ch["strike"].unique())
        near = k[np.abs(np.log(k / ch.attrs["spot"])) <= 0.03]
        assert pd.Series(np.diff(near)).mode().iloc[0] == 100.0  # Bank Nifty strikes are 100 points apart
        assert len(ch) >= 140
    t = days.get_loc(pd.Timestamp("2024-11-12"))
    assert m["chains"][t].attrs["expiry"] == pd.Timestamp("2024-11-13")  # last weekly
    t = days.get_loc(pd.Timestamp("2024-11-14"))
    assert m["chains"][t].attrs["expiry"] == pd.Timestamp("2024-11-27")  # monthly only afterwards
    assert round(m["chains"][t].attrs["T"] * 252) == 7  # 15 and 20 Nov were holidays
    assert m["lot_size"] == 1.0  # keep_units: OI in shares
    fut = m["futures"]
    assert fut["fut_value"].notna().all()
    assert (np.abs(fut["fut_near_settle"] / close.reindex(days).to_numpy() - 1) < 0.03).all()
    for d, tab in m["option_tables"].items():
        assert (tab["strike"] % 100 == 0).all()


@needs_raw
def test_real_banknifty_lot_sizes():
    # Lot sizes printed in the UDiFF files (docs/EVENT_STUDY.md lists the same history): Bank Nifty 15, 30, 35, 30
    for day, bn_lot, nifty_lot in (("20241202", 15, 25), ("20250303", 30, 75), ("20250702", 35, 75),
                                   ("20260102", 30, 65)):
        b = load_bhavcopy_all(RAW / "bhavcopy" / f"fo_{day}.csv.gz")
        o = b[b["instrument"] == "OPT"]
        for sym, lot in (("BANKNIFTY", bn_lot), ("NIFTY", nifty_lot)):
            s = o[o["symbol"] == sym]
            assert set(s.loc[s["expiry"] == s["expiry"].min(), "lot_size"]) == {lot}
    # with OI converted to contracts, the market uses Bank Nifty's lot size
    m = _real("BANKNIFTY", "2025-03-03", "2025-03-07", oi_unit="units")
    assert m["lot_size"] == 30.0


@needs_raw
@pytest.mark.skipif(not (V2 / "daily_gex_panel.csv").exists(), reason="committed Nifty v2 results not present")
def test_real_nifty_daily_panels_match_committed_v2_run(tmp_path):
    """The rebuilt Nifty market gives the committed v2 numbers (window across the July 2024 file-layout switch).

    Rows near the window's edges are skipped: there the window itself changes days-to-expiry counts.
    """
    m = _real("NIFTY", "2024-06-03", "2024-08-30")
    days = pd.DatetimeIndex(m["days"])
    inner = days[(days >= "2024-06-10") & (days <= "2024-08-16")]
    assert len(inner) > 40
    path = tmp_path / "dm.csv"
    rp.write_daily_market(m, path)
    new_dm = pd.read_csv(path, index_col=0, parse_dates=True).loc[inner]
    old_dm = pd.read_csv(V2 / "daily_market.csv", index_col=0, parse_dates=True).loc[inner]
    pd.testing.assert_frame_equal(new_dm, old_dm, check_exact=True)
    new_gex = build_gex_panel(m)
    new_gex.index = pd.DatetimeIndex(new_gex.index)
    path2 = path.with_name("gex.csv")
    new_gex.to_csv(path2)
    new_gex = pd.read_csv(path2, index_col=0, parse_dates=True).loc[inner]
    old_gex = pd.read_csv(V2 / "daily_gex_panel.csv", index_col=0, parse_dates=True).loc[inner]
    pd.testing.assert_frame_equal(new_gex, old_gex, check_exact=True)
