"""Static figures for reports (matplotlib). Colors follow one fixed categorical order."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def scoreboard_bars(sb: pd.DataFrame, group: str, path, title: str):
    """QLIKE relative to mcsGARCH (percent), one bar per model. Negative = better."""
    d = sb[sb["group"] == group].set_index("model")["qlike_vs_mcsgarch_pct"].sort_values()
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(d) + 1.2))
    colors = [ORANGE if m == "mcsgarch" else BLUE for m in d.index]
    ax.barh(d.index, d.values, color=colors, height=0.6)
    for y, v in enumerate(d.values):
        ax.text(v + (0.5 if v >= 0 else -0.5), y, f"{v:+.1f}%", va="center",
                ha="left" if v >= 0 else "right", fontsize=8, color=INK)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_xlabel("QLIKE versus mcsGARCH, % (lower is better)", color=MUTED, fontsize=9)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def gex_timeseries(panel: pd.DataFrame, path, title: str):
    """Net GEX (relative to its 20-day median gross GEX) under each sign rule: small multiples."""
    cols = [c for c in ("net_gex_us_convention", "net_gex_participant", "net_gex_client_mirror", "net_gex_oracle") if c in panel]
    scale = panel["gross_gex"].rolling(20, min_periods=1).median()
    fig, axes = plt.subplots(len(cols), 1, figsize=(8, 1.6 * len(cols) + 0.6), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, cols):
        ax.plot(panel.index, panel[c] / scale, color=BLUE, linewidth=1.2)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.set_ylabel(c.replace("net_gex_", ""), color=MUTED, fontsize=8)
        _style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
    axes[0].set_title(title, loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def diurnal_curves(s: np.ndarray, s_exp: np.ndarray | None, path, title: str):
    fig, ax = plt.subplots(figsize=(7, 3))
    x = np.arange(len(s))
    ax.plot(x, s / s.mean(), color=BLUE, linewidth=2, label="all days")
    if s_exp is not None:
        ax.plot(x, s_exp / s.mean(), color=ORANGE, linewidth=2, label="expiry days")
    ax.set_xlabel("minute of session (0 = 09:15)", color=MUTED, fontsize=9)
    ax.set_ylabel("variance relative to average", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def ablation_dots(abl: pd.DataFrame, family: str, path, title: str, groups=("all", "expiry", "non_expiry")):
    """Change in QLIKE from adding each GEX set (percent; negative = GEX helps), one row per set.

    Filled markers are one-sided Diebold-Mariano p < 0.05, open markers are not.
    """
    d = abl[(abl["family"] == family) & abl["group"].isin(groups)]
    if d.empty:
        return
    labels = list(dict.fromkeys(d["label"]))
    colors = dict(zip(groups, (BLUE, ORANGE, AQUA)))
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(labels) + 1.4))
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
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=28)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
