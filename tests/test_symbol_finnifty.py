"""--symbol FINNIFTY in scripts/run_pipeline.py (pre-registration P104, docs/prereg/P104_THIRD_INDEX_FORECAST.md).

Three promises are tested:

1. The Fin Nifty path reads Fin Nifty's own bars, chains, lot sizes, strike spacing, futures and expiry calendar,
   from its own bhavcopy folder and official-day file, and the "other index expires today" feature
   (``bn_expiry_only``) then refers to Nifty (as for Bank Nifty in P91), not to Bank Nifty.
2. The Nifty and Bank Nifty paths are unchanged when the files also hold Fin Nifty rows (Nifty's other index is
   still Bank Nifty; Bank Nifty's is still Nifty), and their wiring still reads bhavcopy/ and index_daily.csv.
3. On REAL DATA (skipped when the files are absent): calendar facts, chains and price LEVELS only. These tests
   compute no Fin Nifty return-based quantity: phase 1 of P104 forbids looking at any Fin Nifty outcome before
   the pre-registration is committed.
"""

from __future__ import annotations

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

# ---------------------------------------------------------------------------
# Synthetic NSE-layout files holding THREE indices (SYNTHETIC data, made up for the test)
# ---------------------------------------------------------------------------

DAYS = pd.bdate_range("2024-01-01", "2024-02-29")
SPEC = {
    # spot level, strike step, lot size, expiry weekday (Tue = 1, Wed = 2, Thu = 3), futures traded value per day
    "NIFTY": dict(spot=22000.0, step=50.0, lot=25, wd=3, fut_value=1.0e9),
    "BANKNIFTY": dict(spot=48000.0, step=100.0, lot=15, wd=2, fut_value=2.0e9),
    "FINNIFTY": dict(spot=21000.0, step=50.0, lot=40, wd=1, fut_value=3.0e8),
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
    exps = weekly[weekly >= d][:2]
    out = []
    strikes = np.round(spot / s["step"]) * s["step"] + s["step"] * np.arange(-10, 11)
    for e in exps:
        T = max(int(np.busday_count(d.date(), e.date())), 1) / 252
        for j, k in enumerate(strikes):
            for typ in ("CE", "PE"):
                px = float(black76_price(spot, k, T, 0.065, 0.15, typ == "CE"))
                out.append({"TradDt": d.strftime("%Y-%m-%d"), "FinInstrmTp": "IDO", "TckrSymb": sym,
                            "XpryDt": e.strftime("%Y-%m-%d"), "StrkPric": k, "OptnTp": typ, "ClsPric": px,
                            "SttlmPric": px, "UndrlygPric": spot, "OpnIntrst": s["lot"] * (100 + 7 * j + (typ == "PE")),
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
def three_index_files(tmp_path_factory):
    root = tmp_path_factory.mktemp("three_index")
    (root / "bhav_all").mkdir()
    (root / "bhav_two").mkdir()
    for t, d in enumerate(DAYS):
        f = pd.DataFrame(_rows("FINNIFTY", d, t))
        n, b = pd.DataFrame(_rows("NIFTY", d, t)), pd.DataFrame(_rows("BANKNIFTY", d, t))
        name = f"BhavCopy_NSE_FO_0_0_0_{d:%Y%m%d}_F_0000.csv"
        pd.concat([f, b, n]).to_csv(root / "bhav_all" / name, index=False)  # Fin Nifty rows first
        pd.concat([b, n]).to_csv(root / "bhav_two" / name, index=False)
    for sym in SPEC:
        _bars(sym).to_csv(root / f"{sym.lower()}_1min.csv", index=False)
    return root


def _build(root, sym, bhav="bhav_all", **kw):
    kw.setdefault("oi_unit", "units")
    return build_market_from_nse(root / f"{sym.lower()}_1min.csv", root / bhav, symbol=sym, use_file_expiries=True, **kw)


def _in_days(idx):
    return [d for d in pd.DatetimeIndex(idx) if d in set(DAYS)]


def test_synthetic_finnifty_market_uses_fin_nifty_chain_calendar_and_lots(three_index_files):
    m = _build(three_index_files, "FINNIFTY")
    lab = m["labels"]
    days = pd.DatetimeIndex(m["days"])
    assert list(days) == list(DAYS)
    assert list(days[lab["is_expiry"].to_numpy()]) == _in_days(_weekly(1))           # Fin Nifty's Tuesdays
    assert list(days[lab["is_monthly_expiry"].to_numpy()]) == _in_days(_monthly(1))  # its futures expiries
    assert _in_days(m["other_index_expiries"]) == _in_days(_weekly(3))               # other index: Nifty
    for t, d in enumerate(days):
        ch = m["chains"][t]
        assert set(np.diff(np.sort(ch["strike"].unique()))) == {50.0}
        assert ch.attrs["expiry"] == _weekly(1)[_weekly(1) > d][0]
        assert 0.9 < ch.attrs["spot"] / (21000.0 * (1 + 0.001 * t)) < 1.1
    assert m["lot_size"] == 40.0
    assert np.allclose(m["futures"]["fut_value"].to_numpy(), 3.0e8)  # Fin Nifty futures, not another index's
    for tab in m["option_tables"].values():
        assert set(tab["expiry"]) <= set(_weekly(1))
    mk = _build(three_index_files, "FINNIFTY", oi_unit="keep_units")
    assert mk["lot_size"] == 1.0
    assert np.allclose(mk["chains"][5]["oi"].to_numpy(), m["chains"][5]["oi"].to_numpy() * 40)


def test_synthetic_finnifty_other_index_feature_is_nifty_expiry(three_index_files):
    m = _build(three_index_files, "FINNIFTY", oi_unit="keep_units")
    ds = build_dataset(m, build_gex_panel(m), horizon=15)
    day = ds.groupby("date")["bn_expiry_only"].first()
    want = pd.Series([d.weekday() == 3 for d in day.index], index=day.index)  # Nifty Thursday, not a BN Wednesday
    assert day.astype(bool).equals(want)
    assert ds.loc[ds["is_expiry"], "date"].dt.weekday.eq(1).all()


def test_synthetic_nifty_and_banknifty_markets_ignore_fin_nifty_rows(three_index_files):
    for sym, other_wd in (("NIFTY", 2), ("BANKNIFTY", 3)):
        a = _build(three_index_files, sym)
        b = _build(three_index_files, sym, bhav="bhav_two")
        pd.testing.assert_frame_equal(a["labels"], b["labels"])
        for x, y in zip(a["chains"], b["chains"]):
            pd.testing.assert_frame_equal(x, y)
            assert x.attrs == y.attrs
        for d in a["option_tables"]:
            pd.testing.assert_frame_equal(a["option_tables"][d], b["option_tables"][d])
        pd.testing.assert_frame_equal(a["futures"], b["futures"])
        # the other index is unchanged: Bank Nifty for Nifty (original design), Nifty for Bank Nifty (P91)
        assert _in_days(a["other_index_expiries"]) == _in_days(_weekly(other_wd))
        assert list(a["other_index_expiries"]) == list(b["other_index_expiries"])


# ---------------------------------------------------------------------------
# Pipeline wiring (no data needed)
# ---------------------------------------------------------------------------


def _capture_load_market(monkeypatch, tmp_path, **attrs):
    import niftygex.data as data_mod
    pd.DataFrame({"date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                  "index": ["Nifty 50", "Nifty Bank", "India VIX"]}).to_csv(tmp_path / "index_daily.csv", index=False)
    (tmp_path / "finnifty").mkdir(exist_ok=True)
    pd.DataFrame({"date": ["2024-01-04", "2024-01-05"], "index": ["Nifty Financial Services"] * 2}).to_csv(
        tmp_path / "finnifty" / "index_daily_finnifty.csv", index=False)
    got = {}

    def fake(*args, **kw):
        got["args"], got["kw"] = args, kw
        return {}
    monkeypatch.setattr(data_mod, "build_market_from_nse", fake)
    a = types.SimpleNamespace(source="nse", raw=str(tmp_path), oi_unit="keep_units", lot_size=None, rate=0.065,
                              start="2024-01-01", end=None, min_coverage=0.95, **attrs)
    rp.load_market(a)
    return got


def test_load_market_wiring_finnifty(monkeypatch, tmp_path):
    got = _capture_load_market(monkeypatch, tmp_path, symbol="FINNIFTY")
    assert got["args"][0] == tmp_path / "finnifty_1min.csv"
    assert got["args"][1] == tmp_path / "finnifty" / "bhavcopy"
    assert got["kw"]["symbol"] == "FINNIFTY"
    assert list(pd.DatetimeIndex(got["kw"]["official_days"])) == list(pd.to_datetime(["2024-01-04", "2024-01-05"]))
    assert got["kw"]["oi_unit"] == "keep_units" and got["kw"]["use_file_expiries"] is True
    assert "other_symbol" not in got["kw"]  # the default rule gives NIFTY for every index other than Nifty


def test_load_market_wiring_nifty_and_banknifty_unchanged(monkeypatch, tmp_path):
    for sym, bars, day in (("NIFTY", "nifty_1min.csv", "2024-01-01"), ("BANKNIFTY", "banknifty_1min.csv", "2024-01-02")):
        got = _capture_load_market(monkeypatch, tmp_path, symbol=sym)
        assert got["args"][0] == tmp_path / bars
        assert got["args"][1] == tmp_path / "bhavcopy"
        assert list(pd.DatetimeIndex(got["kw"]["official_days"])) == [pd.Timestamp(day)]


# ---------------------------------------------------------------------------
# Real data (skipped when the files are absent). REAL DATA, no Fin Nifty outcome is computed.
# ---------------------------------------------------------------------------

FIN_BHAV = RAW / "finnifty" / "bhavcopy"
FIN_OFFICIAL = RAW / "finnifty" / "index_daily_finnifty.csv"
needs_raw = pytest.mark.skipif(not (RAW / "finnifty_1min.csv").exists() or not FIN_BHAV.exists()
                               or not FIN_OFFICIAL.exists(), reason="real Fin Nifty data not in data/raw")


@pytest.fixture(scope="module")
def fn_real():
    # 21 Oct 2024 to 7 Feb 2025: the last Fin Nifty weeklies, the switch to monthly only, the move of the
    # monthly expiry from the last Tuesday to the last Thursday (January 2025)
    a = types.SimpleNamespace(source="nse", raw=str(RAW), symbol="FINNIFTY", oi_unit="keep_units", lot_size=None,
                              rate=0.065, start="2024-10-21", end="2025-02-07", min_coverage=0.95)
    return rp.load_market(a)


@needs_raw
def test_real_finnifty_expiry_calendar(fn_real):
    m = fn_real
    days = pd.DatetimeIndex(m["days"])
    lab = m["labels"]
    D = pd.to_datetime
    assert list(days[lab["is_expiry"].to_numpy()]) == list(D(["2024-10-22", "2024-10-29", "2024-11-05", "2024-11-12",
                                                                "2024-11-19", "2024-11-26", "2024-12-31",
                                                                "2025-01-30"]))
    assert list(days[lab["is_monthly_expiry"].to_numpy()]) == list(D(["2024-10-29", "2024-11-26", "2024-12-31",
                                                                        "2025-01-30"]))
    other = [d for d in pd.DatetimeIndex(m["other_index_expiries"]) if days[0] <= d <= days[-1]]
    assert len(other) == 16 and all(d.weekday() == 3 for d in other)  # Nifty's Thursdays, not Bank Nifty's
    off = pd.read_csv(FIN_OFFICIAL, parse_dates=["date"])
    assert set(days) <= set(off["date"]) and D("2024-11-01") not in set(days)  # Muhurat session dropped
    assert lab.loc["2024-11-19", "regime"] == "multi_weekly" and lab.loc["2024-11-21", "regime"] == "single_weekly_thu"


@needs_raw
def test_real_finnifty_chains_bars_and_spacing(fn_real):
    m = fn_real
    days = pd.DatetimeIndex(m["days"])
    off = pd.read_csv(FIN_OFFICIAL, parse_dates=["date"]).set_index("date")["close"]
    idx = pd.read_csv(RAW / "index_daily.csv", parse_dates=["date"])
    bank_close = idx[idx["index"] == "Nifty Bank"].set_index("date")["close"]
    assert not m["missing_bhavcopy_days"]
    spacing = []
    for t, d in enumerate(days):
        ch = m["chains"][t]
        # Fin Nifty bars: the last bar is within 1% of NSE's official close (a level, not a return),
        # and far from the Bank Nifty close
        assert abs(ch.attrs["spot_index"] / off[d] - 1) < 0.01
        assert ch.attrs["spot_index"] < 0.6 * bank_close[d]
        assert abs(ch.attrs["spot"] / off[d] - 1) < 0.03
        k = np.sort(ch["strike"].unique())
        near = k[np.abs(np.log(k / ch.attrs["spot"])) <= 0.03]
        spacing.append(pd.Series(np.diff(near)).mode().iloc[0])
    # Fin Nifty strikes are 50 points apart near the money on most days (on 15 of the 754 sample days the
    # nearest chain lists only every 100 points there)
    spacing = pd.Series(spacing)
    assert set(spacing) <= {50.0, 100.0} and spacing.mode().iloc[0] == 50.0
    assert m["lot_size"] == 1.0
    fut = m["futures"]
    assert fut["fut_value"].notna().all()
    assert (np.abs(fut["fut_near_settle"] / off.reindex(days).to_numpy() - 1) < 0.03).all()


@needs_raw
def test_real_finnifty_bhavcopy_folder_matches_p96_files():
    # days inside the P96 window are byte copies of nse_full/fo_index; the Nifty rows equal those of bhavcopy/
    for day in ("20230801", "20241202", "20250829"):
        a = load_bhavcopy_all(FIN_BHAV / f"fo_{day}.csv.gz", symbols=("NIFTY",))
        b = load_bhavcopy_all(RAW / "bhavcopy" / f"fo_{day}.csv.gz", symbols=("NIFTY",))
        pd.testing.assert_frame_equal(a, b)
    for day in ("20230102", "20230731", "20250901", "20260127"):  # downloaded for P104
        f = load_bhavcopy_all(FIN_BHAV / f"fo_{day}.csv.gz", symbols=("FINNIFTY",))
        assert (f["instrument"] == "OPT").sum() > 100 and (f["instrument"] == "FUT").sum() >= 3
        assert pd.Timestamp(f["date"].iloc[0]).normalize() == pd.Timestamp(day)
