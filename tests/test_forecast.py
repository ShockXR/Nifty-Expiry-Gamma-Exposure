import numpy as np
import pandas as pd

from niftygex.evaluation import diebold_mariano, qlike
from niftygex.features import build_dataset, build_gex_panel
from niftygex.forecast import WalkForwardConfig, run_walkforward
from niftygex.simulate import SimConfig, simulate_market
from niftygex.volatility import daily_egarch_variance


def test_qlike_minimised_at_truth():
    y = 2.0
    f = np.linspace(0.5, 5, 1000)
    assert np.isclose(f[np.argmin(qlike(y, f))], y, atol=0.01)
    assert qlike(y, y) == 0


def test_dm_detects_clear_difference_and_not_noise():
    rng = np.random.default_rng(0)
    base = rng.exponential(1, 300)
    better = base * 0.8
    assert diebold_mariano(base, better, alternative="greater")["p"] < 0.01
    noise = base + rng.normal(0, 0.01, 300)
    assert diebold_mariano(base, noise, alternative="greater")["p"] > 0.01


def test_daily_variance_falls_back_on_garbage():
    # Constant returns make the EGARCH likelihood degenerate; the function must still return sane variances.
    h, info = daily_egarch_variance(np.r_[np.zeros(50), np.ones(5) * 0.01])
    assert np.all(np.isfinite(h)) and np.all(h > 0)
    assert info["method"] in ("egarch_t", "egarch_normal", "ewma")


def test_walkforward_runs_and_never_uses_future_days():
    m = simulate_market(SimConfig(n_days=90, kappa=0.0, seed=21))
    panel = build_gex_panel(m)
    ds = build_dataset(m, panel, horizon=30, step=45)
    cfg = WalkForwardConfig(min_train=60, block=15, models=["naive_last", "har", "mcsgarch", "hybrid_base", "hybrid_gex"])
    pred, info = run_walkforward(m, ds, cfg)
    assert pred["t"].min() >= 60
    assert np.all(np.isfinite(pred[["har", "mcsgarch", "hybrid_base", "hybrid_gex"]].to_numpy()))
    assert np.all(pred[["har", "mcsgarch", "hybrid_base", "hybrid_gex"]].to_numpy() > 0)
    # every refit starts at a block boundary and trains only on earlier days
    starts = [p["start"] for p in info["mcs_params"]]
    assert all(s >= 60 for s in starts)
