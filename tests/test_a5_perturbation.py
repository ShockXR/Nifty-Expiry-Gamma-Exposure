"""Tests for experiments/a5_perturbation.py (A5). SYNTHETIC data only: no raw data file is read, no
walk-forward is run. Covers the noise-injection helper (scale, determinism, same draw at both fractions,
untouched flags and NaN cells) and the LightGBM parameter override wrapper."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from niftygex.forecast import WalkForwardConfig, _lgbm_params

_PATH = Path(__file__).resolve().parents[1] / "experiments" / "a5_perturbation.py"
_spec = importlib.util.spec_from_file_location("a5_perturbation", _PATH)
a5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(a5)


def _frame(n=4000, seed=3):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "t": np.repeat(np.arange(n // 20), 20),
        "log_vix": rng.normal(-2.0, 0.3, n),
        "ofi15": rng.normal(0.0, 2.0, n),
        "log_rv_d1": rng.normal(-10.0, 0.7, n),
        "is_expiry": rng.integers(0, 2, n).astype(bool),
        "weekday": rng.integers(0, 5, n).astype(np.int32),
        "y": rng.lognormal(-13.0, 0.5, n),
    })
    df.loc[df.index[:7], "ofi15"] = np.nan          # NaN cells must stay NaN
    return df


def test_noise_columns_are_the_float_inputs_only():
    df = _frame()
    cols = a5.noise_columns(df, features=["is_expiry", "weekday", "log_vix", "ofi15", "log_rv_d1", "not_there"])
    assert cols == ["log_vix", "ofi15", "log_rv_d1"]


def test_noise_scale_determinism_and_untouched_columns():
    df = _frame()
    cols = ["log_vix", "ofi15", "log_rv_d1"]
    train = df["t"].to_numpy() < 100
    out10 = a5.add_input_noise(df, cols, 0.10, 9_900_001, train)
    out10b = a5.add_input_noise(df, cols, 0.10, 9_900_001, train)
    out25 = a5.add_input_noise(df, cols, 0.25, 9_900_001, train)
    out0 = a5.add_input_noise(df, cols, 0.0, 9_900_001, train)
    # same seed, same output; zero fraction, no change; the original frame is not modified
    pd.testing.assert_frame_equal(out10, out10b)
    pd.testing.assert_frame_equal(out0, df)
    assert df["log_vix"].equals(_frame()["log_vix"])
    for c in cols:
        x = df[c].to_numpy(float)
        sd_train = np.nanstd(x[train], ddof=1)
        d10 = out10[c].to_numpy(float) - x
        d25 = out25[c].to_numpy(float) - x
        ok = np.isfinite(x)
        # noise sd close to the requested fraction of the training sd, on train and on test rows alike
        assert abs(np.std(d10[ok & train]) / (0.10 * sd_train) - 1) < 0.08
        assert abs(np.std(d10[ok & ~train]) / (0.10 * sd_train) - 1) < 0.08
        # the 25% run uses the same draw, scaled
        np.testing.assert_allclose(d25[ok], 2.5 * d10[ok], rtol=1e-8, atol=1e-12)  # rounding of (x + noise) - x
        # NaN cells stay NaN
        assert np.array_equal(np.isnan(out10[c].to_numpy(float)), ~ok)
    # flags, codes, the target and identifiers are untouched
    for c in ("is_expiry", "weekday", "y", "t"):
        assert out25[c].equals(df[c]) and out25[c].dtype == df[c].dtype
    # a column with zero training variance is left as it is and does not shift the other draws
    df2 = df.copy()
    df2["flat"] = 1.0
    out_flat = a5.add_input_noise(df2, ["flat"] + cols, 0.10, 9_900_001, train)
    assert out_flat["flat"].equals(df2["flat"])
    out_shift = a5.add_input_noise(df, cols, 0.10, 9_900_001, train)
    assert not np.allclose(out_flat["log_vix"].to_numpy(), out_shift["log_vix"].to_numpy())


def test_lgbm_override_and_registered_values():
    cfg = WalkForwardConfig()
    base = _lgbm_params(cfg, 7, num_leaves=15, min_data_in_leaf=100)
    assert base["learning_rate"] == 0.03 and base["num_leaves"] == 15
    patched = a5.lgbm_override({"learning_rate": 0.015, "num_leaves": 7})
    p = patched(cfg, 7, num_leaves=15, min_data_in_leaf=100)
    assert p["learning_rate"] == 0.015 and p["num_leaves"] == 7 and p["min_data_in_leaf"] == 100
    assert p["seed"] == 7 and p["bagging_fraction"] == base["bagging_fraction"]
    # the registered list, in order, and its concrete values on a 24-anchor dataset (min_leaf 100)
    assert list(a5.PERTURBATIONS) == ["baseline", "lr_half", "lr_double", "leaves_7", "leaves_31", "min_leaf_half",
                                      "min_leaf_double", "noise_10", "noise_25", "block_42", "min_train_250", "single_seed"]
    ds = pd.DataFrame({"minute": np.tile(np.arange(0, 360, 15), 3)})
    s = {n: a5.run_settings(n, ds) for n in a5.PERTURBATIONS}
    assert s["baseline"]["learning_rate"] == 0.03 and s["baseline"]["num_leaves"] == 15 and s["baseline"]["min_data_in_leaf"] == 100
    assert s["baseline"]["block"] == 21 and s["baseline"]["min_train"] == 150 and s["baseline"]["n_seeds"] == 5
    assert s["lr_half"]["learning_rate"] == 0.015 and s["lr_double"]["learning_rate"] == 0.06
    assert s["leaves_7"]["num_leaves"] == 7 and s["leaves_31"]["num_leaves"] == 31
    assert s["min_leaf_half"]["min_data_in_leaf"] == 50 and s["min_leaf_double"]["min_data_in_leaf"] == 200
    assert s["noise_10"]["noise_frac"] == 0.10 and s["noise_25"]["noise_frac"] == 0.25 and s["noise_25"]["noise_seed"] == 9_900_001
    assert s["block_42"]["block"] == 42 and s["min_train_250"]["min_train"] == 250 and s["single_seed"]["n_seeds"] == 1
    assert all(v["seed"] == 7 for v in s.values())
