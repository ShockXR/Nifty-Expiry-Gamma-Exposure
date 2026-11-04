"""Static figures for reports (matplotlib). Colors follow one fixed categorical order.

Colour meaning (kept the same on the demo pages, the deck chart and these figures): BLUE is the
mcsGARCH benchmark, ORANGE is the hybrid (mcsGARCH times a learnt correction), grey is any other
model. The ablation figure uses its own legend (blue, orange, green = all, expiry, non-expiry days).

Every figure carries a data label (REAL DATA or SYNTHETIC) when it can be told: pass ``data_label``
explicitly, or it is inferred from a ``pipeline_nse*`` or ``pipeline_synthetic`` folder in the output path.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID, OTHER = "#1f1f1e", "#6b6a63", "#e4e3dc", "#a9a89f"
GEX_SCALE_WINDOW = 20  # days of the rolling median used to scale net GEX in gex_timeseries


def provenance(path, data_label=None) -> str | None:
    """REAL / SYNTHETIC label for a figure: the explicit label, else read from the output folder name."""
    if data_label:
        return data_label
    parts = [p.lower() for p in Path(str(path)).parts]
    if any(re.fullmatch(r"pipeline_synthetic", p) for p in parts):
        return "SYNTHETIC (simulated market, not real data)"
    if any(re.match(r"pipeline_nse", p) for p in parts):
        return "REAL DATA (NSE)"
    return None


def _footer(fig, lines):
    lines = [x for x in lines if x]
    if lines:
        fig.text(0.01, 0.005, "\n".join(lines), ha="left", va="bottom", fontsize=7.5, color=MUTED)


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def scoreboard_bars(sb: pd.DataFrame, group: str, path, title: str, data_label: str | None = None):
    """QLIKE relative to mcsGARCH (percent), one bar per model. Negative = better.

    Blue is mcsGARCH (the reference, always 0), orange is a hybrid model, grey is any other model.
    Negative values are printed to the right of the zero line so they never run into the model names.
    """
    d = sb[sb["group"] == group].set_index("model")["qlike_vs_mcsgarch_pct"].sort_values()
    days = sb[sb["group"] == group]["days"].max() if "days" in sb else None
    lab = provenance(path, data_label)
    n_dev = sb[sb["group"] == "development"]["days"].max() if "days" in sb and (sb["group"] == "development").any() else None
    n_hold = sb[sb["group"] == "holdout"]["days"].max() if "days" in sb and (sb["group"] == "holdout").any() else None
    foot = [
        (lab + ". " if lab else "") + (f"{int(days)} test days in this group; " if days is not None else "")
        + "walk-forward: each forecast uses only earlier data. Blue = mcsGARCH, orange = hybrid, grey = other.",
        (f"The full test period holds a development part ({int(n_dev)} days) and a hold-out part ({int(n_hold)} days); "
         "see the scoreboard CSV for each separately.") if n_dev and n_hold and group in ("all", "expiry") else "",
    ]
    nfoot = sum(1 for x in foot if x)
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(d) + 1.2 + 0.18 * nfoot))
    colors = [BLUE if m == "mcsgarch" else ORANGE if m.startswith("hybrid") else OTHER for m in d.index]
    ax.barh(d.index, d.values, color=colors, height=0.6)
    span = max(float(d.max()), 0.0) - min(float(d.min()), 0.0)
    pad = 0.012 * span
    for y, v in enumerate(d.values):
        ax.text(v + pad if v >= 0 else pad, y, f"{v:+.1f}%", va="center", ha="left", fontsize=8, color=INK)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_xlim(min(float(d.min()), 0.0) - 0.02 * span, max(float(d.max()), 0.0) + 0.14 * span)
    ax.set_xlabel("QLIKE versus mcsGARCH, % (lower is better)", color=MUTED, fontsize=9)
    ax.set_title(title + (f"  [{lab.split(' (')[0]}]" if lab else ""), loc="left", fontsize=11, color=INK)
    _style(ax)
    fig.tight_layout(rect=(0, 0.012 * (1 + nfoot), 1, 1))
    _footer(fig, foot)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def gex_timeseries(panel: pd.DataFrame, path, title: str, data_label: str | None = None):
    """Net GEX (relative to its 20-day median gross GEX) under each sign rule: small multiples.

    Each panel has its own vertical scale, so compare the shape of the lines, not their height.
    """
    cols = [c for c in ("net_gex_us_convention", "net_gex_participant", "net_gex_client_mirror", "net_gex_oracle") if c in panel]
    scale = panel["gross_gex"].rolling(GEX_SCALE_WINDOW, min_periods=1).median()
    lab = provenance(path, data_label)
    fig, axes = plt.subplots(len(cols), 1, figsize=(8, 1.6 * len(cols) + 0.9), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, cols):
        ax.plot(panel.index, panel[c] / scale, color=BLUE, linewidth=1.2)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.set_ylabel(c.replace("net_gex_", "").replace("_", " "), color=MUTED, fontsize=8)
        _style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
    axes[0].set_title(title + (f"  [{lab.split(' (')[0]}]" if lab else ""), loc="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _footer(fig, [(lab + ". " if lab else "") + "Each panel has its own vertical scale; read the shape and sign, not the height."])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def diurnal_curves(s: np.ndarray, s_exp: np.ndarray | None, path, title: str, data_label: str | None = None):
    """The fitted time-of-day variance curve. The first minutes are far above the rest, so the vertical axis is
    cut where the curve leaves the picture and the cut-off values are printed on the figure."""
    lab = provenance(path, data_label)
    fig, ax = plt.subplots(figsize=(7, 3.3))
    x = np.arange(len(s))
    base = s / s.mean()
    series = [(base, BLUE, "all days")]
    if s_exp is not None:
        series.append((s_exp / s.mean(), ORANGE, "expiry days"))
    allv = np.concatenate([v for v, _, _ in series])
    top = 1.3 * float(np.sort(allv)[-6])
    clipped = top < 0.5 * float(allv.max())
    for v, col, name in series:
        ax.plot(x, v, color=col, linewidth=2, label=name)
    if clipped:
        ax.set_ylim(0, top)
        msg = "; ".join(f"{name} {v[np.argmax(v)]:.1f} at minute {int(np.argmax(v))}" for v, _, name in series if v.max() > top)
        ax.text(0.99, 0.62, "Axis cut: off the scale, " + msg, transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=MUTED)
    ax.set_xlabel("minute of session (0 = 09:15)", color=MUTED, fontsize=9)
    ax.set_ylabel("variance relative to average", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=9, loc="upper right", bbox_to_anchor=(1.0, 0.98))
    ax.set_title(title + (f"  [{lab.split(' (')[0]}]" if lab else ""), loc="left", fontsize=11, color=INK)
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _footer(fig, [(lab + ". " if lab else "") + "Fitted on the whole sample, so this curve is in-sample: it describes the pattern, it is not a forecast test."])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def ablation_dots(abl: pd.DataFrame, family: str, path, title: str, groups=("all", "expiry", "non_expiry"),
                  data_label: str | None = None):
    """Change in QLIKE from adding each GEX set (percent; negative = GEX helps), one row per set.

    Filled markers are one-sided Diebold-Mariano p < 0.05, open markers are not.
    """
    d = abl[(abl["family"] == family) & abl["group"].isin(groups)]
    if d.empty:
        return
    lab = provenance(path, data_label)
    labels = list(dict.fromkeys(d["label"]))
    colors = dict(zip(groups, (BLUE, ORANGE, AQUA)))
    ndays = {g: int(d[d["group"] == g]["days"].max()) for g in groups if "days" in d and (d["group"] == g).any()}
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(labels) + 1.9))
    for k, g in enumerate(groups):
        sub = d[d["group"] == g].set_index("label").reindex(labels)
        y = np.arange(len(labels)) + (k - 1) * 0.22
        sig = sub["p_one_sided"].to_numpy() < 0.05
        ax.scatter(sub["change_pct"][sig], y[sig], color=colors[g], s=36, label=f"{g} (p<0.05)", zorder=3)
        ax.scatter(sub["change_pct"][~sig], y[~sig], facecolors="none", edgecolors=colors[g], s=36, label=g, zorder=3)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("QLIKE change from adding the GEX set, % (negative = helps)", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=46)
    _style(ax)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    days_txt = ", ".join(f"{g}: {n} days" for g, n in ndays.items())
    _footer(fig, [(lab + ". " if lab else "") + "Out-of-sample walk-forward. " + (days_txt + "." if days_txt else ""),
                  "Filled = one-sided Diebold-Mariano p < 0.05 (unadjusted); open = not significant. Placebo = GEX of a random earlier day."])
    fig.savefig(path, dpi=150)
    plt.close(fig)
