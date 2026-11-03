#!/usr/bin/env python
"""Meeting demo (P111): one trading day of the forecasting pipeline, as one self-contained HTML page.

Run from the repository root:

    PYTHONPATH=src OMP_NUM_THREADS=1 python scripts/demo_forecast_day.py --date 2025-12-23
    PYTHONPATH=src OMP_NUM_THREADS=1 python scripts/demo_forecast_day.py --date 2025-12-23 --png
    PYTHONPATH=src OMP_NUM_THREADS=1 python scripts/demo_forecast_day.py --pick

It reads COMMITTED files only and refits nothing:

* ``<run>/forecasts_next15.csv`` (required), ``forecasts_next30.csv`` and ``forecasts_last_hour.csv``
  (optional): one row per (test day, anchor minute) with the realized variance ``y`` of the window
  that starts at the anchor and every model's forecast of it, made at the anchor. The two models
  shown are ``mcsgarch`` (the anchor paper's model) and ``hybrid_base`` (mcsGARCH times a LightGBM
  correction), the pair behind the headline 5.8% and 7.6% gains.
* ``<run>/daily_market.csv`` (optional): the last one-minute price of each day, for the caption.
* ``results/p106_gain_decomposition/shares.csv``, ``contributions.csv`` and ``run_info.json``
  (optional): study P106's group shares of the hybrid's gain, an average over all test days.

``<run>`` is ``results/pipeline_nse_v2`` for NIFTY, ``results/pipeline_nse_banknifty`` for BANKNIFTY
and ``results/pipeline_nse_finnifty`` for FINNIFTY, or the folder given with ``--results``.

It writes ONE HTML file, ``<out>/demo_<index>_<date>.html``, with inline SVG charts, inline CSS and
a few lines of inline JavaScript for the hover tooltip; it needs no network when opened. With
``--png`` it also writes ``<out>/deck_chart_<date>.png`` (1600 x 900 pixels) of the session chart.

QLIKE(y, f) = y / f - log(y / f) - 1, as in ``niftygex.evaluation``. A day's QLIKE is the average
over that day's forecast windows, the project's usual daily loss.

If ``--date`` is not a test day the script prints the nearest test days and exits with status 2
without writing anything. ``--pick`` prints the three walk-through days of ``docs/DEMO.md`` and the
rules that pick them.

Written by an AI agent (Claude) for Shreyan Thanki's project.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

RUN_DIRS = {
    "NIFTY": "results/pipeline_nse_v2",
    "BANKNIFTY": "results/pipeline_nse_banknifty",
    "FINNIFTY": "results/pipeline_nse_finnifty",
}
INDEX_NAMES = {"NIFTY": "Nifty", "BANKNIFTY": "Bank Nifty", "FINNIFTY": "Fin Nifty"}
P106_DIR = "results/p106_gain_decomposition"

MCS, HYB = "mcsgarch", "hybrid_base"
HORIZONS = [
    ("next15", "forecasts_next15.csv", "Next 15 minutes"),
    ("next30", "forecasts_next30.csv", "Next 30 minutes"),
    ("last_hour", "forecasts_last_hour.csv", "Last hour (14:30 to 15:30)"),
]
COLS = ["date", "minute", "horizon", "is_expiry", "tdte", "regime", "y", MCS, HYB]
SESSION_START = 9 * 60 + 15  # 09:15 India time

# Colours: Okabe-Ito-like blue and orange (checked with a colour-vision-deficiency validator:
# protan Delta E 24.7, normal-vision Delta E 33.6), plus a neutral grey for what really happened.
C_MCS, C_HYB = "#2a78d6", "#eb6834"
C_REAL = "#9b9a94"
C_INK, C_INK2, C_MUTED, C_GRID = "#141413", "#45443f", "#6b6a64", "#e4e3de"

# Notes for a few well-known days. These are public news, not computed from the data, and carry
# no numbers; every number on the page is computed from the committed files.
EVENTS = {
    "2024-02-01": "Interim Budget day",
    "2024-06-03": "the first trading day after the general election exit polls; the market rallied",
    "2024-06-04": ("general election results day; the count showed the ruling party short of a "
                   "majority on its own, and the market crashed"),
    "2024-06-05": "the day after the election-result crash",
    "2024-07-23": "Union Budget day (capital gains tax was raised)",
    "2025-04-07": "a global sell-off after new United States tariffs",
    "2025-04-08": "the trading day after the global tariff sell-off of 7 April 2025",
    "2025-05-12": "the first trading day after the India-Pakistan ceasefire; the market rallied",
}
REGIME_WORDS = {
    0: "Thursday expiries, before SEBI's 20 November 2024 change, when several NSE index options expired each week",
    1: "Thursday expiries, after SEBI's 20 November 2024 change left Nifty the only NSE weekly expiry",
    2: "Tuesday expiries, from 1 September 2025",
}
GROUP_LABELS = {
    "realized": "Recent realized volatility",
    "intraday": "Time of day",
    "optionmkt": "Option market + VIX",
    "implied": "Option prices",
    "vix": "India VIX",
    "calendar": "Calendar and expiry",
}
GROUP_ORDER = ["realized", "intraday", "optionmkt", "implied", "vix", "calendar"]


# ----------------------------------------------------------------------------------------------
# Data


def qlike(y, f):
    """Same formula as niftygex.evaluation.qlike (repeated here so the demo needs no package)."""
    y = np.maximum(np.asarray(y, float), 1e-16)
    f = np.maximum(np.asarray(f, float), 1e-16)
    return y / f - np.log(y / f) - 1


def load_forecasts(path: Path) -> pd.DataFrame:
    head = pd.read_csv(path, nrows=0).columns
    missing = [c for c in ("date", "minute", "y", MCS, HYB) if c not in head]
    if missing:
        raise SystemExit(f"{path} lacks the columns {missing}")
    f = pd.read_csv(path, usecols=[c for c in COLS if c in head])
    f["date"] = pd.to_datetime(f["date"]).dt.strftime("%Y-%m-%d")
    if "is_expiry" in f:
        f["is_expiry"] = f["is_expiry"].astype(str).str.lower().isin(["true", "1", "1.0"])
    else:
        f["is_expiry"] = False
    for c in ("tdte", "regime"):
        if c not in f:
            f[c] = np.nan
    f["l_mcs"] = qlike(f["y"], f[MCS])
    f["l_hyb"] = qlike(f["y"], f[HYB])
    return f.sort_values(["date", "minute"]).reset_index(drop=True)


def daily_table(f: pd.DataFrame) -> pd.DataFrame:
    d = f.groupby("date").agg(
        l_mcs=("l_mcs", "mean"), l_hyb=("l_hyb", "mean"), windows=("y", "size"), rv=("y", "sum"),
        is_expiry=("is_expiry", "first"), tdte=("tdte", "first"), regime=("regime", "first"))
    d["gain_pct"] = 100 * (d["l_mcs"] - d["l_hyb"]) / d["l_mcs"]
    first = f[f["minute"] == f["minute"].min()].set_index("date")
    d["open_diff"] = (first["l_mcs"] - first["l_hyb"]).reindex(d.index)
    d["weekday"] = pd.to_datetime(d.index).weekday
    return d


def rank_high(values: pd.Series, v: float) -> int:
    """1 = the largest value."""
    return int((values > v).sum()) + 1


def nearest_days(days: list[str], date: str, k: int = 3) -> tuple[list[str], list[str]]:
    s = sorted(days)
    before = [x for x in s if x < date][-k:]
    after = [x for x in s if x > date][:k]
    return before, after


def run_dir(index: str, results: str | None) -> Path:
    if results:
        return Path(results)
    return ROOT / RUN_DIRS[index]


# ----------------------------------------------------------------------------------------------
# Formatting


def fmt_q(x: float) -> str:
    return f"{x:.3f}"


def fmt_var(v: float) -> str:
    """A variance in units of 10^-6, three significant figures."""
    u = v * 1e6
    if u >= 1000:
        return f"{u:,.0f}"
    if u >= 100:
        return f"{u:.0f}"
    return f"{u:#.3g}".rstrip(".")


def fmt_pct(x: float, signed: bool = False) -> str:
    return f"{x:+.1f}%" if signed else f"{x:.1f}%"


def ordinal(n: int) -> str:
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n:,}{suf}"


def clock(minute: float) -> str:
    m = SESSION_START + int(minute)
    return f"{m // 60:02d}:{m % 60:02d}"


def long_date(date: str) -> str:
    t = pd.Timestamp(date)
    return f"{t.day_name()} {t.day} {t.month_name()} {t.year}"


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def nice_step(span: float, n: int = 5) -> float:
    raw = span / max(n, 1)
    if raw <= 0 or not math.isfinite(raw):
        return 1.0
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if m * mag >= raw:
            return m * mag
    return 10 * mag


def tick_text(v: float) -> str:
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    s = f"{v:.4f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def badge(x: float, y: float, label: str, anchor: str = "end") -> str:
    """The data label (REAL DATA or SYNTHETIC) drawn inside an SVG, top right."""
    w = 14 * len(label) + 24
    x0 = x - w if anchor == "end" else x
    return (f'<g class="data-label"><rect x="{x0:.0f}" y="{y:.0f}" width="{w}" height="34" rx="6" '
            f'fill="{C_INK}"/><text x="{x0 + w / 2:.0f}" y="{y + 24:.0f}" text-anchor="middle" '
            f'font-size="19" font-weight="700" fill="#ffffff" letter-spacing="1">{esc(label)}</text></g>')


# ----------------------------------------------------------------------------------------------
# Charts (inline SVG)


def svg_session(day: pd.DataFrame, label: str, horizon_min: int) -> str:
    """Bars: realized variance of each window. Lines: the two forecasts made at the window's start.
    A strip under the axis says which forecast had the lower QLIKE in each window."""
    W, H = 1400, 660
    x0, x1, y0, y1 = 130, 1375, 92, 470
    n = len(day)
    slot = (x1 - x0) / n
    bw = min(24.0, slot * 0.6)
    y_u = day["y"].to_numpy() * 1e6
    m_u = day[MCS].to_numpy() * 1e6
    h_u = day[HYB].to_numpy() * 1e6
    top = float(np.nanmax(np.concatenate([y_u, m_u, h_u])))
    step = nice_step(top * 1.05)
    ymax = math.ceil(top * 1.05 / step) * step
    ys = lambda v: y1 - (v / ymax) * (y1 - y0)  # noqa: E731
    xc = lambda i: x0 + slot * (i + 0.5)  # noqa: E731

    out = [f'<svg class="chart session" viewBox="0 0 {W} {H}" role="img" '
           f'aria-labelledby="sess-title" xmlns="http://www.w3.org/2000/svg" font-family="inherit">',
           f'<title id="sess-title">{esc(label)}: realized {horizon_min}-minute variance in each window '
           'against the mcsGARCH and hybrid forecasts made at the start of the window</title>',
           f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>']
    # legend
    lx = x0
    out.append(f'<rect x="{lx}" y="16" width="26" height="30" rx="3" fill="{C_REAL}"/>'
               f'<text x="{lx + 36}" y="40" font-size="25" fill="{C_INK}">Realized (what happened)</text>')
    lx += 385
    out.append(f'<line x1="{lx}" y1="31" x2="{lx + 52}" y2="31" stroke="{C_MCS}" stroke-width="4" '
               f'stroke-dasharray="10 6"/><circle cx="{lx + 26}" cy="31" r="7" fill="{C_MCS}" '
               f'stroke="#fff" stroke-width="2"/><text x="{lx + 64}" y="40" font-size="25" '
               f'fill="{C_INK}">mcsGARCH forecast</text>')
    lx += 335
    out.append(f'<line x1="{lx}" y1="31" x2="{lx + 52}" y2="31" stroke="{C_HYB}" stroke-width="4"/>'
               f'<rect x="{lx + 19}" y="24" width="14" height="14" fill="{C_HYB}" stroke="#fff" '
               f'stroke-width="2"/><text x="{lx + 64}" y="40" font-size="25" fill="{C_INK}">'
               'Hybrid forecast</text>')
    out.append(badge(W - 25, 12, label))
    # grid and y axis
    v = 0.0
    while v <= ymax + 1e-12:
        yy = ys(v)
        out.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x1}" y2="{yy:.1f}" stroke="{C_GRID}" stroke-width="1"/>'
                   f'<text x="{x0 - 12}" y="{yy + 7:.1f}" text-anchor="end" font-size="23" '
                   f'fill="{C_INK2}">{tick_text(v)}</text>')
        v += step
    out.append(f'<text transform="translate(34 {(y0 + y1) / 2:.0f}) rotate(-90)" text-anchor="middle" '
               f'font-size="23" fill="{C_INK2}">Variance (\u00d7 10\u207b\u2076)</text>')
    # bars
    for i in range(n):
        h = y1 - ys(y_u[i])
        out.append(f'<path d="{bar_path(xc(i) - bw / 2, y1, bw, h)}" fill="{C_REAL}"/>')
    # lines
    pts_m = " ".join(f"{xc(i):.1f},{ys(m_u[i]):.1f}" for i in range(n))
    pts_h = " ".join(f"{xc(i):.1f},{ys(h_u[i]):.1f}" for i in range(n))
    out.append(f'<polyline points="{pts_m}" fill="none" stroke="{C_MCS}" stroke-width="4" '
               'stroke-dasharray="10 6" stroke-linejoin="round" stroke-linecap="round"/>')
    out.append(f'<polyline points="{pts_h}" fill="none" stroke="{C_HYB}" stroke-width="4" '
               'stroke-linejoin="round" stroke-linecap="round"/>')
    for i in range(n):
        out.append(f'<circle cx="{xc(i):.1f}" cy="{ys(m_u[i]):.1f}" r="6" fill="{C_MCS}" '
                   'stroke="#fff" stroke-width="2"/>')
        out.append(f'<rect x="{xc(i) - 6:.1f}" y="{ys(h_u[i]) - 6:.1f}" width="12" height="12" '
                   f'fill="{C_HYB}" stroke="#fff" stroke-width="2"/>')
    # x axis
    out.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="{C_INK2}" stroke-width="1.5"/>')
    minutes = day["minute"].to_numpy()
    every = max(1, int(round(60 / max(np.diff(minutes).min(), 1)))) if n > 1 else 1
    for i in range(n):
        if minutes[i] % 60 == 0 or (n <= 6) or (every == 1):
            out.append(f'<line x1="{x0 + slot * i:.1f}" y1="{y1}" x2="{x0 + slot * i:.1f}" y2="{y1 + 8}" '
                       f'stroke="{C_INK2}" stroke-width="1.5"/><text x="{x0 + slot * i:.1f}" y="{y1 + 34}" '
                       f'text-anchor="middle" font-size="23" fill="{C_INK2}">{clock(minutes[i])}</text>')
    out.append(f'<text x="{(x0 + x1) / 2:.0f}" y="{y1 + 68}" text-anchor="middle" font-size="23" '
               f'fill="{C_INK2}">Start of the {horizon_min}-minute window (India time)</text>')
    # strip: which forecast was closer
    sy = y1 + 110
    out.append(f'<text x="{x0}" y="{sy - 12}" font-size="23" fill="{C_INK}">Closer forecast in each '
               'window (lower QLIKE): <tspan font-weight="700">M</tspan> = mcsGARCH, '
               '<tspan font-weight="700">H</tspan> = hybrid</text>')
    lm, lh = day["l_mcs"].to_numpy(), day["l_hyb"].to_numpy()
    for i in range(n):
        if lh[i] < lm[i]:
            fill, letter = C_HYB, "H"
        elif lm[i] < lh[i]:
            fill, letter = C_MCS, "M"
        else:
            fill, letter = C_REAL, "="
        out.append(f'<rect x="{x0 + slot * i + 2:.1f}" y="{sy}" width="{slot - 4:.1f}" height="38" rx="4" '
                   f'fill="{fill}"/><text x="{xc(i):.1f}" y="{sy + 27}" text-anchor="middle" font-size="23" '
                   f'font-weight="700" fill="#ffffff">{letter}</text>')
    # hover layer
    out.append(f'<line class="xhair" x1="0" y1="{y0}" x2="0" y2="{sy + 38}" stroke="{C_INK}" '
               'stroke-width="1.5" visibility="hidden"/>')
    for i in range(n):
        out.append(f'<rect class="hit" data-i="{i}" x="{x0 + slot * i:.1f}" y="{y0}" width="{slot:.1f}" '
                   f'height="{sy + 38 - y0}" fill="transparent" tabindex="0"/>')
    out.append("</svg>")
    return "\n".join(out)


def bar_path(x: float, base: float, w: float, h: float, r: float = 4.0) -> str:
    """A column with a rounded top and a square base."""
    if h <= 0:
        return f"M{x:.1f},{base:.1f}h{w:.1f}"
    r = min(r, w / 2, h)
    top = base - h
    return (f"M{x:.1f},{base:.1f}V{top + r:.1f}Q{x:.1f},{top:.1f} {x + r:.1f},{top:.1f}"
            f"H{x + w - r:.1f}Q{x + w:.1f},{top:.1f} {x + w:.1f},{top + r:.1f}V{base:.1f}Z")


def svg_gain_hist(daily: pd.DataFrame, date: str, label: str) -> str:
    """Histogram of the hybrid's daily gain over all test days, with the chosen day marked."""
    W, H = 760, 470
    x0, x1, y0, y1 = 86, 740, 90, 330
    fs = 23
    g = daily["gain_pct"].to_numpy()
    lo = max(-60.0, math.floor(np.nanmin(g) / 5) * 5)
    hi = min(60.0, math.ceil(np.nanmax(g) / 5) * 5)
    if hi <= lo:
        hi = lo + 5
    edges = np.arange(lo, hi + 5, 5.0)
    gc = np.clip(g, lo, hi - 1e-9)
    counts, _ = np.histogram(gc, bins=edges)
    v = float(daily.loc[date, "gain_pct"])
    k = int(np.clip(np.searchsorted(edges, min(max(v, lo), hi - 1e-9), side="right") - 1, 0, len(counts) - 1))
    cmax = max(int(counts.max()), 1)
    step = nice_step(cmax, 4)
    top = math.ceil(cmax / step) * step
    xs = lambda val: x0 + (val - lo) / (hi - lo) * (x1 - x0)  # noqa: E731
    ys = lambda c: y1 - c / top * (y1 - y0)  # noqa: E731
    out = [f'<svg class="chart hist" viewBox="0 0 {W} {H}" role="img" aria-labelledby="hist-title" '
           'xmlns="http://www.w3.org/2000/svg" font-family="inherit">',
           f'<title id="hist-title">{esc(label)}: the hybrid\'s gain over mcsGARCH on each test day</title>',
           f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>', badge(W - 10, 6, label)]
    c = 0.0
    while c <= top + 1e-9:
        out.append(f'<line x1="{x0}" y1="{ys(c):.1f}" x2="{x1}" y2="{ys(c):.1f}" stroke="{C_GRID}"/>'
                   f'<text x="{x0 - 10}" y="{ys(c) + 8:.1f}" text-anchor="end" font-size="{fs}" '
                   f'fill="{C_INK2}">{int(c)}</text>')
        c += step
    bw = (x1 - x0) / len(counts)
    for i, cnt in enumerate(counts):
        col = C_INK if i == k else "#c9c8c1"
        out.append(f'<path d="{bar_path(x0 + bw * i + 1, y1, bw - 2, y1 - ys(cnt))}" fill="{col}"/>')
    zx = xs(0.0)
    if x0 <= zx <= x1:
        out.append(f'<line x1="{zx:.1f}" y1="{y0 - 10}" x2="{zx:.1f}" y2="{y1}" stroke="{C_INK2}" '
                   'stroke-width="2"/>')
    # the chosen day
    mx = x0 + bw * (k + 0.5)
    my = max(ys(counts[k]) - 14, y0 - 16)
    out.append(f'<text x="{mx:.1f}" y="{my:.1f}" text-anchor="middle" font-size="{fs}" '
               f'font-weight="700" fill="{C_INK}" stroke="#ffffff" stroke-width="6" paint-order="stroke">this day</text>')
    t = math.ceil(lo / 10) * 10 if (hi - lo) > 40 else lo
    while t <= hi + 1e-9:
        out.append(f'<text x="{xs(t):.1f}" y="{y1 + 32}" text-anchor="middle" font-size="{fs}" '
                   f'fill="{C_INK2}">{int(t)}%</text>')
        t += 20 if (hi - lo) > 80 else (10 if (hi - lo) > 40 else 5)
    out.append(f'<text x="{x0}" y="{y1 + 72}" font-size="{fs}" fill="{C_INK2}">&#8592; hybrid worse</text>'
               f'<text x="{x1}" y="{y1 + 72}" text-anchor="end" font-size="{fs}" fill="{C_INK2}">'
               'hybrid better &#8594;</text>')
    out.append(f'<text x="{(x0 + x1) / 2:.0f}" y="{y1 + 112}" text-anchor="middle" font-size="{fs - 2}" '
               f'fill="{C_INK2}">Hybrid\'s QLIKE below mcsGARCH\'s that day (%)</text>')
    out.append(f'<text transform="translate(24 {(y0 + y1) / 2:.0f}) rotate(-90)" text-anchor="middle" '
               f'font-size="{fs - 2}" fill="{C_INK2}">Test days</text>')
    out.append("</svg>")
    return "\n".join(out)


def load_p106(p106_dir: Path, index: str) -> dict | None:
    sp, cp = p106_dir / "shares.csv", p106_dir / "contributions.csv"
    if not sp.exists():
        return None
    sh = pd.read_csv(sp)
    sh = sh[sh["index"] == index]
    if sh.empty:
        return None
    out = {"shares": sh, "contrib": None, "counts": {}, "days": None, "supported": set()}
    if cp.exists():
        c = pd.read_csv(cp)
        c = c[(c["index"] == index) & (c["subset"] == "all days")
              & c["seed"].astype(str).str.contains("headline")]
        out["contrib"] = c
        d = c.loc[c["target"] == "next15", "days"]
        out["days"] = int(d.iloc[0]) if len(d) else None
    rp = p106_dir / "run_info.json"
    if rp.exists():
        info = json.loads(rp.read_text())
        for key in ("drop_sets", "groups"):
            for g, cols in (info.get(key) or {}).items():
                out["counts"].setdefault(g, len(cols))
    vp = p106_dir / "verdict.csv"
    if vp.exists() and index == "NIFTY":  # the registered verdicts are for Nifty only
        v = pd.read_csv(vp)
        v = v[(v["part"] == "P1") & (v["verdict"] == "supported")]
        both = v.groupby("group")["target"].nunique()
        out["supported"] = set(both[both >= 2].index)
    return out


def svg_p106(p: dict, label: str) -> str:
    sh = p["shares"]
    rows = []
    for g in GROUP_ORDER:
        r15 = sh[(sh["target"] == "next15") & (sh["variant"] == g)]
        if r15.empty:
            continue
        r30 = sh[(sh["target"] == "next30") & (sh["variant"] == g)]
        rows.append((g, 100 * r15.iloc[0]["share_of_gain"], 100 * r15.iloc[0]["share_ci_low"],
                     100 * r15.iloc[0]["share_ci_high"],
                     100 * r30.iloc[0]["share_of_gain"] if len(r30) else None))
    W = 760
    rh = 72
    x0, x1 = 300, 650
    y0 = 104
    fs = 23
    yb = y0 + rh * len(rows)
    H = yb + 120
    lo = min([0.0] + [r[2] for r in rows]) - 2
    hi = max([0.0] + [r[3] for r in rows]) + 2
    step = nice_step(hi - lo, 3)
    lo, hi = math.floor(lo / step) * step, math.ceil(hi / step) * step
    xs = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)  # noqa: E731
    out = [f'<svg class="chart p106" viewBox="0 0 {W} {H}" role="img" aria-labelledby="p106-title" '
           'xmlns="http://www.w3.org/2000/svg" font-family="inherit">',
           f'<title id="p106-title">{esc(label)}: study P106, share of the hybrid\'s gain carried by each '
           'group of inputs, average over all test days</title>',
           f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>', badge(W - 10, 6, label),
           f'<text x="{x0}" y="{y0 - 30}" font-size="{fs - 2}" fill="{C_INK2}">15-minute forecasts</text>',
           f'<text x="{W - 6}" y="{y0 - 30}" text-anchor="end" font-size="{fs - 2}" fill="{C_INK2}">30 min</text>']
    t = lo
    while t <= hi + 1e-9:
        out.append(f'<line x1="{xs(t):.1f}" y1="{y0 - 12}" x2="{xs(t):.1f}" y2="{yb}" stroke="{C_GRID}"/>'
                   f'<text x="{xs(t):.1f}" y="{yb + 30}" text-anchor="middle" font-size="{fs - 2}" '
                   f'fill="{C_INK2}">{tick_text(t)}%</text>')
        t += step
    zx = xs(0.0)
    out.append(f'<line x1="{zx:.1f}" y1="{y0 - 12}" x2="{zx:.1f}" y2="{yb}" stroke="{C_INK2}" stroke-width="2"/>')
    for j, (g, s, clo, chi, s30) in enumerate(rows):
        yc = y0 + rh * j + rh / 2
        n_in = p["counts"].get(g)
        name = GROUP_LABELS.get(g, g)
        out.append(f'<text x="{x0 - 14}" y="{yc - 1:.1f}" text-anchor="end" font-size="{fs}" '
                   f'fill="{C_INK}">{esc(name)}</text>')
        if n_in:
            out.append(f'<text x="{x0 - 14}" y="{yc + 23:.1f}" text-anchor="end" font-size="{fs - 4}" '
                       f'fill="{C_MUTED}">{n_in} input{"s" if n_in != 1 else ""}</text>')
        bx, bw = (zx, xs(s) - zx) if s >= 0 else (xs(s), zx - xs(s))
        col = C_INK2 if g in p.get("supported", set()) else "#b4b3ac"
        out.append(f'<rect x="{bx:.1f}" y="{yc - 13:.1f}" width="{max(bw, 1):.1f}" height="26" rx="3" '
                   f'fill="{col}"/>')
        out.append(f'<line x1="{xs(clo):.1f}" y1="{yc:.1f}" x2="{xs(chi):.1f}" y2="{yc:.1f}" '
                   f'stroke="{C_INK}" stroke-width="2"/>'
                   f'<line x1="{xs(clo):.1f}" y1="{yc - 10:.1f}" x2="{xs(clo):.1f}" y2="{yc + 10:.1f}" '
                   f'stroke="{C_INK}" stroke-width="2"/>'
                   f'<line x1="{xs(chi):.1f}" y1="{yc - 10:.1f}" x2="{xs(chi):.1f}" y2="{yc + 10:.1f}" '
                   f'stroke="{C_INK}" stroke-width="2"/>')
        tx = xs(chi) + 8
        out.append(f'<text x="{tx:.1f}" y="{yc + 8:.1f}" font-size="{fs}" font-weight="700" '
                   f'fill="{C_INK}">{s:.0f}%</text>')
        if s30 is not None:
            out.append(f'<text x="{W - 6}" y="{yc + 8:.1f}" text-anchor="end" font-size="{fs}" '
                       f'fill="{C_INK2}">{s30:.0f}%</text>')
    out.append(f'<text x="10" y="{yb + 72}" font-size="{fs - 3}" fill="{C_INK2}">Bar: share of the gain lost '
               'without the group. Whiskers: 95% interval.</text>')
    if p.get("supported"):
        out.append(f'<text x="10" y="{yb + 102}" font-size="{fs - 3}" fill="{C_INK2}">Dark bars: groups that '
                   'pass P106\'s registered test at both horizons.</text>')
    out.append("</svg>")
    return "\n".join(out)


# ----------------------------------------------------------------------------------------------
# Caption and page


def caption(date: str, drow: pd.Series, daily: pd.DataFrame, market: pd.DataFrame | None,
            index_name: str, note: str | None) -> str:
    """One or two plain sentences: what kind of day it was, how far the index moved, how busy it was.
    The date itself is the page's heading, so it is not repeated here."""
    if note:
        what = note
    elif date in EVENTS:
        what = EVENTS[date]
    elif bool(drow["is_expiry"]):
        monthly = False
        if market is not None and "is_monthly_expiry" in market and date in market.index:
            monthly = str(market.loc[date, "is_monthly_expiry"]).lower() in ("true", "1")
        what = f"a {index_name} {'monthly' if monthly else 'weekly'} expiry day"
        reg = drow.get("regime")
        if pd.notna(reg) and int(reg) in REGIME_WORDS and index_name == "Nifty":
            what += f" ({REGIME_WORDS[int(reg)]})"
    else:
        td = drow.get("tdte")
        if pd.notna(td) and int(td) == 1:
            what = f"an ordinary trading day, the day before a {index_name} expiry"
        elif pd.notna(td) and int(td) > 1:
            what = f"an ordinary trading day, {int(td)} trading days before the next {index_name} expiry"
        else:
            what = "an ordinary trading day"
    s = what[0].upper() + what[1:] + "."
    bits = []
    if market is not None and date in market.index and "close_last_bar" in market:
        pos = market.index.get_loc(date)
        if pos > 0:
            prev = float(market["close_last_bar"].iloc[pos - 1])
            cur = float(market.loc[date, "close_last_bar"])
            if prev > 0 and np.isfinite(cur):
                ch = 100 * (cur / prev - 1)
                if abs(ch) < 0.05:
                    bits.append(f"{index_name} ended the day almost unchanged")
                else:
                    word = "higher" if ch > 0 else "lower"
                    bits.append(f"{index_name} ended the day {abs(ch):.1f}% {word} (last one-minute price "
                                "against the day before)")
    med = float(daily["rv"].median())
    if med > 0:
        ratio = float(drow["rv"]) / med
        calm = ("much busier" if ratio > 3 else "busier" if ratio > 1.25
                else "calmer" if ratio < 0.8 else "about as busy")
        bits.append(f"the session was {calm} than usual (realized variance {ratio:.1f} times the "
                    "median test day's)")
    if bits:
        s += " " + bits[0][0].upper() + bits[0][1:] + (", and " + bits[1] if len(bits) > 1 else "") + "."
    return s


CSS = """
:root {
  color-scheme: light;
  --surface: #f7f6f2; --card: #ffffff; --ink: #141413; --ink2: #45443f; --muted: #6b6a64;
  --line: #e4e3de; --mcs: #2a78d6; --hyb: #eb6834; --real: #9b9a94;
}
* { box-sizing: border-box; }
html { font-size: 20px; }
body { margin: 0; background: var(--surface); color: var(--ink);
  font-family: "Segoe UI", "Helvetica Neue", Arial, "Noto Sans", sans-serif; line-height: 1.45; }
main { max-width: 1500px; margin: 0 auto; padding: 24px 16px 48px; }
.eyebrow { font-size: 1rem; color: var(--ink2); letter-spacing: 0.02em; }
h1 { font-size: 2.2rem; margin: 4px 0 8px; line-height: 1.15; }
h2 { font-size: 1.35rem; margin: 0 0 6px; }
.caption { font-size: 1.25rem; margin: 0 0 20px; max-width: 70em; }
.tag { display: inline-block; background: var(--ink); color: #fff; font-weight: 700; font-size: 0.85rem;
  letter-spacing: 0.06em; padding: 3px 10px; border-radius: 6px; vertical-align: middle; margin-left: 8px; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 20px;
  margin: 0 0 20px; }
.sub { color: var(--ink2); margin: 0 0 10px; font-size: 1.05rem; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px;
  margin: 0 0 20px; }
.tile { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 16px 20px; }
.tile .lab { font-size: 1.05rem; color: var(--ink2); display: flex; align-items: center; gap: 10px; }
.tile .val { font-size: 2.4rem; font-weight: 700; line-height: 1.1; margin: 6px 0 4px; }
.tile .rank { font-size: 1rem; color: var(--ink2); }
.sw { display: inline-block; width: 28px; height: 6px; border-radius: 3px; flex: none; }
.sw.mcs { background: repeating-linear-gradient(90deg, var(--mcs) 0 9px, transparent 9px 14px); height: 5px; }
.sw.hyb { background: var(--hyb); }
.cols { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 20px; }
.cols .card { margin: 0; }
.chartwrap { position: relative; }
svg.chart { width: 100%; height: auto; display: block; }
.warn { background: #fff4d6; border-left: 6px solid #c98500; padding: 8px 12px; margin: 0 0 10px;
  font-weight: 600; }
table { border-collapse: collapse; width: 100%; font-size: 1rem; }
th, td { text-align: right; padding: 6px 10px; border-bottom: 1px solid var(--line); white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
th { color: var(--ink2); font-weight: 600; }
.scroll { overflow-x: auto; }
details summary { cursor: pointer; font-weight: 600; font-size: 1.1rem; }
.note { color: var(--ink2); font-size: 0.95rem; }
footer { color: var(--muted); font-size: 0.9rem; margin-top: 12px; }
code { font-size: 0.9em; }
#tip { position: absolute; pointer-events: none; background: #141413; color: #fff; border-radius: 8px;
  padding: 10px 14px; font-size: 1rem; line-height: 1.4; min-width: 260px; display: none; z-index: 5;
  box-shadow: 0 4px 14px rgba(0,0,0,0.25); }
#tip b { font-weight: 700; }
#tip .k { display: inline-block; width: 12px; height: 12px; border-radius: 2px; margin-right: 6px; }
.hit:focus { outline: 3px solid #141413; }
@media (max-width: 700px) { html { font-size: 16px; } .cols { grid-template-columns: 1fr; } }
"""

JS = """
(function () {
  var data = JSON.parse(document.getElementById('win-data').textContent);
  var wrap = document.getElementById('sess-wrap');
  var tip = document.getElementById('tip');
  var svg = wrap.querySelector('svg');
  var xh = svg.querySelector('.xhair');
  function show(el, evt) {
    var d = data[+el.getAttribute('data-i')];
    var x = +el.getAttribute('x') + (+el.getAttribute('width')) / 2;
    xh.setAttribute('x1', x); xh.setAttribute('x2', x); xh.setAttribute('visibility', 'visible');
    tip.innerHTML = '<b>' + d.w + '</b><br>' +
      'Realized: <b>' + d.y + '</b><br>' +
      '<span class="k" style="background:#2a78d6"></span>mcsGARCH: <b>' + d.m + '</b> (QLIKE ' + d.lm + ')<br>' +
      '<span class="k" style="background:#eb6834"></span>Hybrid: <b>' + d.h + '</b> (QLIKE ' + d.lh + ')<br>' +
      'LightGBM correction: \u00d7 ' + d.c + '<br>Closer: <b>' + d.win + '</b>';
    tip.style.display = 'block';
    var r = wrap.getBoundingClientRect(), b = el.getBoundingClientRect();
    var left = b.left - r.left + b.width + 12;
    if (left + tip.offsetWidth > r.width) left = b.left - r.left - tip.offsetWidth - 12;
    tip.style.left = Math.max(0, left) + 'px';
    tip.style.top = '70px';
  }
  function hide() { tip.style.display = 'none'; xh.setAttribute('visibility', 'hidden'); }
  svg.querySelectorAll('.hit').forEach(function (el) {
    el.addEventListener('mouseenter', function (e) { show(el, e); });
    el.addEventListener('focus', function (e) { show(el, e); });
    el.addEventListener('mouseleave', hide);
    el.addEventListener('blur', hide);
  });
})();
"""


def horizon_rows(run: Path, date: str, primary: tuple[str, pd.DataFrame]) -> list[dict]:
    rows = []
    for key, fname, title in HORIZONS:
        if key == primary[0]:
            f = primary[1]
        else:
            path = run / fname
            if not path.exists():
                continue
            f = load_forecasts(path)
        d = daily_table(f)
        if date not in d.index:
            continue
        r = d.loc[date]
        rows.append({
            "key": key, "title": title, "windows": int(r["windows"]), "days": len(d),
            "l_mcs": float(r["l_mcs"]), "l_hyb": float(r["l_hyb"]), "gain": float(r["gain_pct"]),
            "rank_mcs": rank_high(d["l_mcs"], r["l_mcs"]), "rank_hyb": rank_high(d["l_hyb"], r["l_hyb"]),
            "rank_gain": rank_high(d["gain_pct"], r["gain_pct"]),
            "all_gain": 100 * (d["l_mcs"].mean() - d["l_hyb"].mean()) / d["l_mcs"].mean(),
            "wins": int((d["l_hyb"] < d["l_mcs"]).sum()),
        })
    return rows


def build_page(date: str, index: str, run: Path, f: pd.DataFrame, daily: pd.DataFrame,
               market: pd.DataFrame | None, p106: dict | None, label: str, note: str | None) -> tuple[str, dict]:
    name = INDEX_NAMES.get(index, index.title())
    day = f[f["date"] == date].reset_index(drop=True)
    r = daily.loc[date]
    nd = len(daily)
    hmin = int(day["horizon"].iloc[0]) if "horizon" in day and pd.notna(day["horizon"].iloc[0]) else 15
    rank_m = rank_high(daily["l_mcs"], r["l_mcs"])
    rank_h = rank_high(daily["l_hyb"], r["l_hyb"])
    rank_g = rank_high(daily["gain_pct"], r["gain_pct"])
    all_gain = 100 * (daily["l_mcs"].mean() - daily["l_hyb"].mean()) / daily["l_mcs"].mean()
    n_hyb_closer = int((day["l_hyb"] < day["l_mcs"]).sum())
    n_days_won = int((daily["l_hyb"] < daily["l_mcs"]).sum())
    med_gain = float(daily["gain_pct"].median())
    day_no = int(list(daily.index).index(date)) + 1
    cap = caption(date, r, daily, market, name, note)
    hrows = horizon_rows(run, date, ("next15", f))

    win = []
    for _, w in day.iterrows():
        closer = "hybrid" if w["l_hyb"] < w["l_mcs"] else ("mcsGARCH" if w["l_mcs"] < w["l_hyb"] else "tie")
        win.append({"w": f"{clock(w['minute'])} to {clock(w['minute'] + hmin)}", "y": fmt_var(w["y"]),
                    "m": fmt_var(w[MCS]), "h": fmt_var(w[HYB]), "lm": fmt_q(w["l_mcs"]),
                    "lh": fmt_q(w["l_hyb"]), "c": f"{w[HYB] / w[MCS]:.2f}", "win": closer})

    gain_word = "lower" if r["gain_pct"] >= 0 else "higher"
    tiles = f"""
<section class="tiles" aria-label="The day's QLIKE">
  <div class="tile"><div class="lab"><span class="sw mcs"></span>mcsGARCH (the paper's model): day's QLIKE</div>
    <div class="val">{fmt_q(r['l_mcs'])}</div>
    <div class="rank">{ordinal(rank_m)} highest of {nd:,} test days (1st = hardest day)</div></div>
  <div class="tile"><div class="lab"><span class="sw hyb"></span>Hybrid (mcsGARCH \u00d7 LightGBM correction): day's QLIKE</div>
    <div class="val">{fmt_q(r['l_hyb'])}</div>
    <div class="rank">{ordinal(rank_h)} highest of {nd:,} test days</div></div>
  <div class="tile"><div class="lab">Hybrid's loss compared with mcsGARCH's, this day</div>
    <div class="val">{abs(r['gain_pct']):.1f}% {gain_word}</div>
    <div class="rank">{ordinal(rank_g)} best of {nd:,} days for the hybrid. Over all {nd:,} days: {all_gain:.1f}% lower.</div></div>
</section>"""

    hist = svg_gain_hist(daily, date, label)
    sess = svg_session(day, label, hmin)
    if p106 is not None:
        days_txt = f"{p106['days']:,}" if p106.get("days") else "the"
        foot = []
        sh = p106["shares"]
        main5 = sh[(sh["target"] == "next15") & sh["variant"].isin(["implied", "vix", "intraday", "calendar", "realized"])]
        if len(main5) == 5:
            foot.append(f"The five separate groups' shares add up to {100 * main5['share_of_gain'].sum():.0f}% "
                        "at 15 minutes; the rest of the gain is shared, because when one group is removed others "
                        "can partly stand in for it.")
        c = p106.get("contrib")
        if c is not None:
            mi = c[(c["target"] == "next15") & (c["variant"] == "mcsinfo")]
            if len(mi):
                foot.append(f"A hybrid that sees only what mcsGARCH sees (past one-minute returns, the time of day "
                            f"and the expiry-day flag) is {abs(mi.iloc[0]['variant_margin_vs_mcsgarch_pct']):.1f}% "
                            f"better than mcsGARCH at 15 minutes; the full hybrid is "
                            f"{abs(mi.iloc[0]['hybrid_margin_vs_mcsgarch_pct']):.1f}% better.")
        p106_html = f"""
  <div class="card">
    <h2>Where the hybrid's gain comes from (study P106)</h2>
    <p class="warn">An AVERAGE over all {days_txt} test days, not this day.</p>
    <p class="sub">P106 refitted the hybrid without one group of inputs at a time and measured how much of
    the gain was lost.</p>
    {svg_p106(p106, label)}
    {''.join(f'<p class="note">{esc(t)}</p>' for t in foot)}
  </div>"""
    else:
        p106_html = """
  <div class="card"><h2>Where the hybrid's gain comes from (study P106)</h2>
    <p class="note">The P106 files were not found, so this panel is empty.</p></div>"""

    htable = "".join(
        f"<tr><td>{esc(h['title'])}</td><td>{h['windows']}</td><td>{fmt_q(h['l_mcs'])} "
        f"({ordinal(h['rank_mcs'])})</td><td>{fmt_q(h['l_hyb'])} ({ordinal(h['rank_hyb'])})</td>"
        f"<td>{fmt_pct(-h['gain'], True)} ({ordinal(h['rank_gain'])})</td><td>{fmt_pct(-h['all_gain'], True)}</td></tr>"
        for h in hrows)
    wtable = "".join(
        f"<tr><td>{w['w']}</td><td>{w['y']}</td><td>{w['m']}</td><td>{w['h']}</td><td>{w['c']}</td>"
        f"<td>{w['lm']}</td><td>{w['lh']}</td><td>{w['win']}</td></tr>" for w in win)

    open_txt = ""
    if len(day):
        w0 = day.iloc[0]
        better = "the hybrid" if w0["l_hyb"] < w0["l_mcs"] else "mcsGARCH"
        open_txt = (f" In the first window ({clock(w0['minute'])}) {better} was closer: QLIKE "
                    f"{fmt_q(w0['l_mcs'])} for mcsGARCH and {fmt_q(w0['l_hyb'])} for the hybrid.")

    src = [str((run / "forecasts_next15.csv").relative_to(ROOT)) if run.is_relative_to(ROOT) else str(run / "forecasts_next15.csv")]
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Forecast demo {esc(date)}</title>
<style>{CSS}</style>
</head>
<body>
<main>
<div class="eyebrow">{esc(name)} volatility forecasts, one test day (day {day_no:,} of {nd:,}) <span class="tag">{esc(label)}</span></div>
<h1>{esc(long_date(date))}</h1>
<p class="caption" id="caption">{esc(cap)}</p>
{tiles}
<section class="card">
  <h2>What happened, and what each model forecast</h2>
  <p class="sub">Grey bars: the realized variance of each {hmin}-minute window (the sum of squared one-minute
  returns). Lines: the forecast of that window, made at its start. The hybrid was closer in
  {n_hyb_closer} of {len(day)} windows.{esc(open_txt)} Hover over a window for its numbers.</p>
  <div class="chartwrap" id="sess-wrap">{sess}<div id="tip" role="status"></div></div>
</section>
<section class="cols">
  <div class="card">
    <h2>How unusual was this day?</h2>
    <p class="sub">Each bar counts test days. The hybrid beat mcsGARCH on {n_days_won:,} of {nd:,} days; on the
    median day its QLIKE was {abs(med_gain):.1f}% {"lower" if med_gain >= 0 else "higher"}. The black bar holds this day.</p>
    {hist}
  </div>
  {p106_html}
</section>
<section class="card">
  <h2>The same day at every forecast horizon</h2>
  <p class="sub">QLIKE averaged over the day's forecast windows; lower is better. In brackets: the day's rank
  among all test days (1st = highest loss, or for the gain, 1st = the hybrid's best day).</p>
  <div class="scroll"><table>
  <thead><tr><th>Horizon</th><th>Windows</th><th>mcsGARCH QLIKE</th><th>Hybrid QLIKE</th>
  <th>Hybrid vs mcsGARCH</th><th>All test days</th></tr></thead>
  <tbody>{htable}</tbody></table></div>
  <p class="note">Negative = the hybrid's loss is lower. "All test days" compares the average QLIKE over all
  days, the project's headline measure.</p>
</section>
<section class="card">
<details>
  <summary>Every window of the day (table view of the chart)</summary>
  <div class="scroll"><table>
  <thead><tr><th>Window</th><th>Realized (\u00d7 10\u207b\u2076)</th><th>mcsGARCH (\u00d7 10\u207b\u2076)</th><th>Hybrid (\u00d7 10\u207b\u2076)</th>
  <th>Correction</th><th>QLIKE mcsGARCH</th><th>QLIKE hybrid</th><th>Closer</th></tr></thead>
  <tbody>{wtable}</tbody></table></div>
  <p class="note">Correction = hybrid / mcsGARCH, the factor LightGBM multiplied mcsGARCH's forecast by.</p>
</details>
</section>
<section class="card note">
  <b>How to read QLIKE.</b> QLIKE(y, f) = y / f - log(y / f) - 1 scores a variance forecast f against the
  realized variance y. It is 0 for a perfect forecast and grows as the forecast misses; forecasting too low is
  punished more than forecasting too high. Both models were fitted only on days before each test day
  (walk-forward), so every forecast here was made without seeing the future.
</section>
<footer>
  Data: {esc(label)}. Source files: <code>{esc(src[0])}</code> and the other horizons' files in the same folder,
  <code>daily_market.csv</code> (caption) and <code>{esc(P106_DIR)}/</code> (P106 panel). The models are the
  columns <code>{MCS}</code> and <code>{HYB}</code>. Nothing was refitted for this page. Made with
  <code>scripts/demo_forecast_day.py</code>, written by an AI agent (Claude).
</footer>
</main>
<script type="application/json" id="win-data">{json.dumps(win)}</script>
<script>{JS}</script>
</body>
</html>
"""
    summary = {"date": date, "caption": cap, "l_mcs": float(r["l_mcs"]), "l_hyb": float(r["l_hyb"]),
               "gain_pct": float(r["gain_pct"]), "rank_mcs": rank_m, "rank_hyb": rank_h, "rank_gain": rank_g,
               "days": nd, "hyb_closer": n_hyb_closer, "windows": len(day), "horizons": hrows}
    return page, summary


# ----------------------------------------------------------------------------------------------
# Deck PNG


def write_png(day: pd.DataFrame, date: str, summary: dict, path: Path, label: str, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
    fig.patch.set_facecolor("white")
    n = len(day)
    x = np.arange(n)
    ax.bar(x, day["y"] * 1e6, width=0.6, color=C_REAL, zorder=2)
    ax.plot(x, day[MCS] * 1e6, color=C_MCS, lw=4, ls=(0, (4, 2.5)), marker="o", ms=10, mec="white", mew=2, zorder=3)
    ax.plot(x, day[HYB] * 1e6, color=C_HYB, lw=4, marker="s", ms=9, mec="white", mew=2, zorder=4)
    mins = day["minute"].to_numpy()
    ticks = [i for i in range(n) if mins[i] % 60 == 0]
    ax.set_xticks([t - 0.5 for t in ticks], [clock(mins[t]) for t in ticks], fontsize=20)
    ax.set_xlim(-0.7, n - 0.3)
    ax.tick_params(axis="y", labelsize=20)
    ax.set_ylabel("Variance of the 15-minute window (\u00d7 10\u207b\u2076)", fontsize=20, color=C_INK2)
    ax.set_xlabel("Start of the 15-minute window (India time)", fontsize=20, color=C_INK2)
    ax.grid(axis="y", color=C_GRID, lw=1, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    handles = [Patch(color=C_REAL, label="Realized (what happened)"),
               Line2D([], [], color=C_MCS, lw=4, ls=(0, (4, 2.5)), marker="o", ms=10, mec="white",
                      label="mcsGARCH (paper's model)"),
               Line2D([], [], color=C_HYB, lw=4, marker="s", ms=9, mec="white",
                      label="Hybrid (mcsGARCH × LightGBM)")]
    fig.legend(handles=handles, loc="upper left", fontsize=20, frameon=False, ncol=3,
               bbox_to_anchor=(0.055, 0.815), handlelength=2.2, columnspacing=1.2)
    gain_word = "lower" if summary["gain_pct"] >= 0 else "higher"
    fig.text(0.06, 0.935, title, fontsize=30, fontweight="bold", color=C_INK, ha="left", va="baseline")
    line = (f"Day's QLIKE (lower is better): mcsGARCH {summary['l_mcs']:.3f}, hybrid {summary['l_hyb']:.3f} "
            f"({abs(summary['gain_pct']):.1f}% {gain_word}).")
    w0 = day.iloc[0]
    line2 = (f"First window ({clock(w0['minute'])}): mcsGARCH {w0['l_mcs']:.3f}, hybrid {w0['l_hyb']:.3f}. "
             f"The hybrid was closer in {summary['hyb_closer']} of {summary['windows']} windows.")
    fig.text(0.06, 0.885, line, fontsize=21, color=C_INK2, ha="left", va="baseline")
    fig.text(0.06, 0.845, line2, fontsize=21, color=C_INK2, ha="left", va="baseline")
    fig.text(0.97, 0.935, f" {label} ", fontsize=22, fontweight="bold", color="white", ha="right",
             va="baseline", bbox=dict(boxstyle="round,pad=0.35", fc=C_INK, ec=C_INK))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.74, bottom=0.12)
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------------------------------
# Picking the walk-through days


def pick_days(daily: pd.DataFrame) -> list[dict]:
    """The three days of docs/DEMO.md, each by a fixed rule (median rules take the lower middle)."""
    picks = []
    dd = pd.to_datetime(daily.index)
    tue = daily[daily["is_expiry"] & (daily["weekday"] == 1) & (dd >= pd.Timestamp("2025-10-01"))]
    if len(tue):
        s = tue.sort_values(["gain_pct"], kind="mergesort")
        day = s.index[(len(s) - 1) // 2]
        picks.append({"role": "ordinary Tuesday expiry", "date": day,
                      "rule": (f"Nifty expiry days on a Tuesday from 1 October 2025 on ({len(s)} days); the one "
                               "with the median daily QLIKE gain of the hybrid over mcsGARCH")})
    crash = "2024-06-04"
    worst = (daily["l_hyb"] - daily["l_mcs"]).idxmax()
    if crash in daily.index:
        r = int((daily["l_hyb"] - daily["l_mcs"]).rank(ascending=False, method="min").loc[crash])
        picks.append({"role": "election-result crash", "date": crash,
                      "rule": (f"named in advance; by the hybrid's daily QLIKE minus mcsGARCH's it is the "
                               f"{ordinal(r)} worst day for the hybrid (the worst is {worst})")})
    else:
        picks.append({"role": "worst day for the hybrid", "date": worst,
                      "rule": "largest daily QLIKE of the hybrid minus mcsGARCH's"})
    taken = {p["date"] for p in picks}
    op = daily[(~daily["is_expiry"]) & (daily["open_diff"] > 0) & (~daily.index.isin(taken))]
    if len(op):
        s = op.sort_values("open_diff", kind="mergesort")
        day = s.index[(len(s) - 1) // 2]
        picks.append({"role": "hybrid wins at the open", "date": day,
                      "rule": (f"non-expiry days on which the hybrid's 09:15 forecast had the lower QLIKE "
                               f"({len(s)} days); the one with the median 09:15 QLIKE gain")})
    return picks


# ----------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--date", help="test day, YYYY-MM-DD")
    g.add_argument("--pick", action="store_true", help="print the three walk-through days and their rules")
    ap.add_argument("--index", default="NIFTY", choices=sorted(RUN_DIRS), help="index (default NIFTY)")
    ap.add_argument("--out", default=str(ROOT / "results" / "demo"), help="output folder (default results/demo)")
    ap.add_argument("--results", default=None, help="folder with forecasts_*.csv (default: the committed run)")
    ap.add_argument("--p106", default=str(ROOT / P106_DIR), help="folder with the P106 shares and contributions")
    ap.add_argument("--label", default="REAL DATA", help="data label printed on every chart")
    ap.add_argument("--note", default=None, help="replace the caption's description of the day")
    ap.add_argument("--png", action="store_true", help="also write deck_chart_<date>.png (1600 x 900)")
    a = ap.parse_args(argv)

    t0 = time.time()
    run = run_dir(a.index, a.results)
    path = run / "forecasts_next15.csv"
    if not path.exists():
        print(f"No forecasts file at {path}", file=sys.stderr)
        return 2
    f = load_forecasts(path)
    daily = daily_table(f)

    if a.pick:
        for p in pick_days(daily):
            r = daily.loc[p["date"]]
            print(f"{p['role']}: {p['date']}  QLIKE mcsGARCH {fmt_q(r['l_mcs'])}, hybrid {fmt_q(r['l_hyb'])}, "
                  f"gain {fmt_pct(r['gain_pct'], True)}")
            print(f"    rule: {p['rule']}")
        return 0

    try:
        date = pd.Timestamp(a.date).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        print(f"Cannot read the date {a.date!r}; use YYYY-MM-DD.", file=sys.stderr)
        return 2
    if date not in daily.index:
        before, after = nearest_days(list(daily.index), date)
        print(f"{date} is not a test day for {a.index} (test days run from {daily.index.min()} to "
              f"{daily.index.max()}, {len(daily)} days).")
        print("Nearest test days before: " + (", ".join(before) if before else "none"))
        print("Nearest test days after:  " + (", ".join(after) if after else "none"))
        return 2

    market = None
    mp = run / "daily_market.csv"
    if mp.exists():
        market = pd.read_csv(mp)
        market["date"] = pd.to_datetime(market["date"]).dt.strftime("%Y-%m-%d")
        market = market.set_index("date").sort_index()
    p106 = load_p106(Path(a.p106), a.index)

    page, summary = build_page(date, a.index, run, f, daily, market, p106, a.label, a.note)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    html_path = out / f"demo_{a.index.lower()}_{date}.html"
    html_path.write_text(page, encoding="utf-8")
    print(f"Wrote {html_path} ({html_path.stat().st_size / 1024:.0f} KB)")
    print(f"Caption: {summary['caption']}")
    for h in summary["horizons"]:
        print(f"  {h['title']}: QLIKE mcsGARCH {fmt_q(h['l_mcs'])} ({ordinal(h['rank_mcs'])} highest of "
              f"{h['days']}), hybrid {fmt_q(h['l_hyb'])} ({ordinal(h['rank_hyb'])}), hybrid vs mcsGARCH "
              f"{fmt_pct(-h['gain'], True)} ({ordinal(h['rank_gain'])} best day for the hybrid)")
    print(f"  Hybrid closer in {summary['hyb_closer']} of {summary['windows']} 15-minute windows.")
    if a.png:
        name = INDEX_NAMES.get(a.index, a.index)
        title = f"{name}, {long_date(date)}"
        day = f[f["date"] == date].reset_index(drop=True)
        png_path = out / f"deck_chart_{date}.png"
        write_png(day, date, summary, png_path, a.label, title)
        print(f"Wrote {png_path}")
    print(f"Done in {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
