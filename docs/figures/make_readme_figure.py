"""Draw the README's one-day forecast figure from committed forecasts only.

It reads results/pipeline_nse_v2/forecasts_next15.csv (the walk-forward forecasts behind the
Nifty results) for one test day, 23 December 2025, the day docs/DEMO.md picks by a fixed rule
(the median Tuesday expiry since October 2025 by the day's gain). It refits nothing. Run from the
repository root:

    python docs/figures/make_readme_figure.py

It writes docs/figures/nifty_2025-12-23_forecasts.png. The figure shows one day only and
deliberately prints no score: day-level scores are not the 792-day results.
"""

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results" / "pipeline_nse_v2" / "forecasts_next15.csv"
OUT = ROOT / "docs" / "figures" / "nifty_2025-12-23_forecasts.png"
DATE = "2025-12-23"

INK = "#1d2433"
INK2 = "#5b6475"
GRID = "#e6e8ee"
REAL = "#c9ccd3"
MCS = "#1f4e8c"
HYB = "#c2571a"


def clock(minute: int) -> str:
    h, m = divmod(9 * 60 + 15 + int(minute), 60)
    return f"{h:02d}:{m:02d}"


def main() -> None:
    f = pd.read_csv(SRC, usecols=["date", "minute", "is_expiry", "y", "mcsgarch", "hybrid_base"])
    day = f[f["date"] == DATE].sort_values("minute").reset_index(drop=True)
    assert len(day) == 25 and day["is_expiry"].all(), (len(day), day["is_expiry"].unique())
    x = range(len(day))

    plt.rcParams.update({"font.family": "Inter", "font.size": 13, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig, ax = plt.subplots(figsize=(5.6, 4.6), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    ax.bar(x, day["y"] * 1e6, width=0.7, color=REAL, zorder=2)
    ax.plot(x, day["mcsgarch"] * 1e6, color=MCS, lw=2.2, ls=(0, (4, 2)), marker="o", ms=4.5, mec="white",
            mew=0.8, zorder=3)
    ax.plot(x, day["hybrid_base"] * 1e6, color=HYB, lw=2.2, marker="s", ms=4.5, mec="white", mew=0.8, zorder=4)

    # direct labels instead of a legend
    ax.annotate("realized", xy=(0, day["y"].iloc[0] * 1e6), xytext=(1.8, 2.85), color=INK2, fontsize=12.5,
                arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.8})
    ax.text(8.5, 0.62, "mcsGARCH (our version),\ndashed", color=MCS, fontsize=12.5, fontweight="semibold")
    ax.annotate("hybrid, solid", xy=(18, day["hybrid_base"].iloc[18] * 1e6), xytext=(13.2, 1.55), color=HYB,
                fontsize=12.5, fontweight="semibold", arrowprops={"arrowstyle": "-", "color": HYB, "lw": 0.8})

    ticks = [i for i in x if day["minute"].iloc[i] % 120 == 0]
    ax.set_xticks(ticks, [clock(day["minute"].iloc[i]) for i in ticks])
    ax.set_xlim(-0.7, len(day) - 0.3)
    ax.set_ylim(0, 3.3)
    ax.set_ylabel("15-minute variance (× 10⁻⁶)", fontsize=12)
    ax.set_xlabel("start of the 15-minute window (IST)", fontsize=12)
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("Nifty 50, Tuesday 23 December 2025 (expiry)", loc="left", color=INK, fontsize=14,
                 fontweight="semibold", pad=8)
    fig.text(0.02, 0.012, "Real data, one test day picked by a fixed rule; each forecast\nis made at the "
             "start of its window", color=INK2, fontsize=10.5, va="bottom")
    fig.subplots_adjust(left=0.12, right=0.98, top=0.92, bottom=0.235)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor="white")
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
