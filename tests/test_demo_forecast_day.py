"""Tests for scripts/demo_forecast_day.py (P111 meeting demo). SYNTHETIC data only.

Every test writes a tiny synthetic forecasts file (and, where needed, synthetic P106 files) into
tmp_path. Nothing here reads the repository's results or raw data.
"""

import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from niftygex.evaluation import qlike as qlike_pkg

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "demo_forecast_day.py"
_spec = importlib.util.spec_from_file_location("demo_forecast_day", _PATH)
demo = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = demo
_spec.loader.exec_module(demo)

DAYS = ["2025-10-06", "2025-10-07", "2025-10-08", "2025-10-13", "2025-10-14", "2025-10-15"]
EXPIRY = {"2025-10-07", "2025-10-14"}
MINUTES = [0, 15, 30, 45, 60]
DASHES = (chr(0x2013), chr(0x2014))  # en dash, em dash (written as code points, not characters)


def _forecasts(days=DAYS, minutes=MINUTES, horizon=15, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for t, d in enumerate(days):
        for m in minutes:
            y = float(rng.lognormal(0, 0.6)) * 1e-6
            rows.append({"t": 150 + t, "date": d, "minute": m, "horizon": horizon, "is_expiry": d in EXPIRY,
                         "tdte": 0.0 if d in EXPIRY else 1.0, "regime": 2, "y": y,
                         "naive_last": y * 1.1,
                         "mcsgarch": y * float(rng.lognormal(0, 0.4)),
                         "hybrid_base": y * float(rng.lognormal(0, 0.3))})
    return pd.DataFrame(rows)


@pytest.fixture
def run(tmp_path):
    """A synthetic run folder: three forecasts files and a daily market file."""
    r = tmp_path / "run"
    r.mkdir()
    _forecasts().to_csv(r / "forecasts_next15.csv", index=False)
    _forecasts(minutes=[0, 15, 30, 45], horizon=30, seed=1).to_csv(r / "forecasts_next30.csv", index=False)
    _forecasts(minutes=[315], horizon=60, seed=2).to_csv(r / "forecasts_last_hour.csv", index=False)
    mk = pd.DataFrame({"date": ["2025-10-03"] + DAYS,
                       "is_monthly_expiry": [False] * 7,
                       "close_last_bar": [100.0, 101.0, 99.0, 99.0, 102.0, 100.0, 100.5]})
    mk.to_csv(r / "daily_market.csv", index=False)
    return r


@pytest.fixture
def p106(tmp_path):
    """Synthetic P106 files with the committed files' columns."""
    p = tmp_path / "p106"
    p.mkdir()
    groups = ["implied", "vix", "intraday", "calendar", "realized", "optionmkt", "mcsinfo"]
    shares = []
    for tgt in ("next15", "next30"):
        for i, g in enumerate(groups[:6]):
            s = 0.05 * (i - 2)
            shares.append({"index": "NIFTY", "target": tgt, "variant": g, "share_of_gain": s,
                           "share_ci_low": s - 0.1, "share_ci_high": s + 0.1})
    pd.DataFrame(shares).to_csv(p / "shares.csv", index=False)
    contrib = [{"index": "NIFTY", "target": tgt, "subset": "all days", "seed": "seed set 7 (headline)",
                "variant": g, "days": len(DAYS), "contribution_points": 0.1, "dm_stat": 0.0, "p_one_sided": 0.5,
                "hybrid_margin_vs_mcsgarch_pct": -4.0, "variant_margin_vs_mcsgarch_pct": -4.5}
               for tgt in ("next15", "next30") for g in groups]
    pd.DataFrame(contrib).to_csv(p / "contributions.csv", index=False)
    (p / "run_info.json").write_text(json.dumps({"groups": {"realized": ["a", "b"], "vix": ["c"]},
                                                 "drop_sets": {"optionmkt": ["c", "d", "e"]}}))
    pd.DataFrame([{"part": "P1", "target": t, "group": "realized", "verdict": "supported"}
                  for t in ("next15", "next30")]).to_csv(p / "verdict.csv", index=False)
    return p


def _main(args):
    return demo.main([str(a) for a in args])


def test_qlike_matches_the_package():
    y = np.array([1e-6, 2e-6, 5e-7])
    f = np.array([2e-6, 2e-6, 1e-6])
    np.testing.assert_allclose(demo.qlike(y, f), qlike_pkg(y, f))


def test_html_has_the_expected_elements(run, p106, tmp_path):
    out = tmp_path / "out"
    assert _main(["--date", "2025-10-07", "--results", run, "--out", out, "--p106", p106,
                  "--label", "SYNTHETIC"]) == 0
    files = list(out.iterdir())
    assert [x.name for x in files] == ["demo_nifty_2025-10-07.html"]
    html = files[0].read_text(encoding="utf-8")

    # one self-contained file: no external scripts, stylesheets, images or fonts
    assert not re.search(r'(src|href)\s*=\s*"(https?:)?//', html)
    assert "<link" not in html and "@import" not in html
    assert not any(d in html for d in DASHES)

    # (a) the session chart, (b) the histogram, (d) the P106 panel; every chart carries the data label
    svgs = re.findall(r"<svg.*?</svg>", html, flags=re.S)
    assert len(svgs) == 3
    assert all("SYNTHETIC" in s and 'class="data-label"' in s for s in svgs)
    assert "REAL DATA" not in html
    assert 'class="chart session"' in html and 'class="chart hist"' in html and 'class="chart p106"' in html

    # (a) one bar, two forecast markers and one hover target per window
    sess = svgs[0]
    assert sess.count('class="hit"') == len(MINUTES)
    assert sess.count("<circle") == len(MINUTES) + 1  # plus the legend marker
    assert "Realized (what happened)" in sess and "mcsGARCH forecast" in sess and "Hybrid forecast" in sess
    win = json.loads(re.search(r'<script type="application/json" id="win-data">(.*?)</script>', html,
                               flags=re.S).group(1))
    assert [w["w"][:5] for w in win] == ["09:15", "09:30", "09:45", "10:00", "10:15"]

    # (b) the day's QLIKE for each model, computed as the mean over the day's windows
    f = _forecasts()
    d = f[f["date"] == "2025-10-07"]
    lm = qlike_pkg(d["y"], d["mcsgarch"]).mean()
    lh = qlike_pkg(d["y"], d["hybrid_base"]).mean()
    assert f">{lm:.3f}<" in html and f">{lh:.3f}<" in html
    assert f"of {len(DAYS)} test days" in html
    for title in ("Next 15 minutes", "Next 30 minutes", "Last hour (14:30 to 15:30)"):
        assert title in html

    # (c) the caption: an expiry day, and the move from the market file (101 the day before, 99 that day)
    cap = re.search(r'<p class="caption" id="caption">(.*?)</p>', html, flags=re.S).group(1)
    assert "weekly expiry day" in cap and "2.0% lower" in cap

    # (d) P106 is labelled as an average over all test days, not this day
    assert "not this day" in html
    assert "Recent realized volatility" in svgs[2] and "Time of day" in svgs[2]
    assert "Dark bars" in svgs[2]


def test_not_a_test_day_prints_neighbours_and_writes_nothing(run, p106, tmp_path, capsys):
    out = tmp_path / "out"
    assert _main(["--date", "2025-10-10", "--results", run, "--out", out, "--p106", p106]) == 2
    text = capsys.readouterr().out
    assert "not a test day" in text
    assert "2025-10-08" in text and "2025-10-13" in text
    assert not out.exists()


def test_bad_date_string_exits_cleanly(run, tmp_path):
    assert _main(["--date", "not-a-date", "--results", run, "--out", tmp_path / "o"]) == 2


def test_missing_p106_and_market_files_still_render(run, tmp_path):
    (run / "daily_market.csv").unlink()
    out = tmp_path / "out"
    assert _main(["--date", "2025-10-08", "--results", run, "--out", out, "--p106", tmp_path / "none"]) == 0
    html = (out / "demo_nifty_2025-10-08.html").read_text(encoding="utf-8")
    assert "P106 files were not found" in html
    assert html.count("<svg") == 2
    assert "REAL DATA" in html  # the default label
    assert "ended the day" not in html  # no market file, so no move in the caption


def test_png_is_1600_by_900(run, p106, tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib.image as mpimg

    out = tmp_path / "out"
    assert _main(["--date", "2025-10-14", "--results", run, "--out", out, "--p106", p106, "--png"]) == 0
    img = mpimg.imread(out / "deck_chart_2025-10-14.png")
    assert img.shape[:2] == (900, 1600)


def test_pick_days_rules():
    dates = ["2024-06-03", "2024-06-04", "2025-09-30", "2025-10-07", "2025-10-14", "2025-10-21", "2025-10-22",
             "2025-10-23", "2025-10-24"]
    daily = pd.DataFrame({
        "is_expiry": [False, False, True, True, True, True, False, False, False],
        "gain_pct": [1.0, -2.0, 50.0, 10.0, -5.0, 3.0, 0.0, 0.0, 0.0],
        "l_mcs": [0.2, 3.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
        "l_hyb": [0.2, 3.1, 0.1, 0.18, 0.21, 0.19, 0.5, 0.2, 0.2],
        "open_diff": [0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.2, 0.4, -0.1],
    }, index=dates)
    daily["weekday"] = pd.to_datetime(daily.index).weekday
    picks = {p["role"]: p["date"] for p in demo.pick_days(daily)}
    # Tuesday expiries from October 2025 (gains 10, -5, 3): the median is 3; 30 September is excluded
    assert picks["ordinary Tuesday expiry"] == "2025-10-21"
    assert picks["election-result crash"] == "2024-06-04"
    # non-expiry days winning at the open, other than 4 June 2024: 0.3, 0.2, 0.4 -> median 0.3
    assert picks["hybrid wins at the open"] == "2024-06-03"
    # without 4 June 2024 the second day is the hybrid's worst day
    picks2 = {p["role"]: p["date"] for p in demo.pick_days(daily.drop(index="2024-06-04"))}
    assert picks2["worst day for the hybrid"] == "2025-10-22"


def test_ranks_and_nearest_days():
    s = pd.Series([0.1, 0.5, 0.3, 0.9])
    assert demo.rank_high(s, 0.9) == 1 and demo.rank_high(s, 0.1) == 4
    before, after = demo.nearest_days(["2025-01-01", "2025-01-03", "2025-01-06"], "2025-01-04", k=2)
    assert before == ["2025-01-01", "2025-01-03"] and after == ["2025-01-06"]
    assert demo.clock(0) == "09:15" and demo.clock(375) == "15:30"
    assert demo.ordinal(1) == "1st" and demo.ordinal(12) == "12th" and demo.ordinal(23) == "23rd"
