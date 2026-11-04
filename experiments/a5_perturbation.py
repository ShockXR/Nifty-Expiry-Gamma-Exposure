"""A5: perturbation reruns of the forecasting hybrid (registered plan: docs/prereg/A5_PERTURBATION.md).

Written by an AI agent (Agent C of the backtest and stress pass, 4 November 2026). The question: is the
hybrid's QLIKE gain over mcsGARCH (results/pipeline_nse_v2, headline 7.65% at next30) fragile to settings
nobody tuned on the test days? Each perturbation refits ONLY mcsGARCH (deterministic, cached in memory
across runs) and hybrid_base on the same cached dataset, with base seed 7, exactly as P99
(experiments/seed_bootstrap.py) does, and compares the two with the repo's Diebold-Mariano test on daily
average losses. LightGBM settings are changed by wrapping niftygex.forecast._lgbm_params for the duration
of one run; nothing under src/ is modified.

    PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/a5_perturbation.py --target next30 \
        --perturbation all --out results/a5_perturbation > results/a5_perturbation/run_stdout_next30.txt 2>&1

Outputs (REAL DATA with --source nse): <out>/daily_losses_<target>_<name>.csv (date, qlike_hybrid,
qlike_mcsgarch), one row per run appended to <out>/summary.csv, <out>/run_info.json. Forecasts of each run
go to --work (default data/processed/a5_perturbation, git-ignored); a finished run is read back, not
rerun. ``--source synthetic`` runs the same code on simulated data (tests and smoke runs only; label
SYNTHETIC). Test: tests/test_a5_perturbation.py (synthetic, no raw data).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from niftygex import forecast as fc  # noqa: E402
from niftygex.evaluation import daily_losses, diebold_mariano, qlike  # noqa: E402
from niftygex.features import BASE_FEATURES, GEX_SETS, build_dataset, build_gex_panel  # noqa: E402
from niftygex.implied import build_implied_panel  # noqa: E402
from niftygex.volatility import McsGarch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sb = _load("seed_bootstrap", ROOT / "experiments" / "seed_bootstrap.py")   # P99: config, loader, repro check
rp = sb.rp                                                                 # scripts/run_pipeline.py

HEADLINE_RUN = sb.HEADLINE_RUN
HEADLINE_SEED = sb.HEADLINE_SEED        # 7
MODELS = sb.MODELS                      # ["mcsgarch", "hybrid_base"]
NOISE_SEED = 9_900_001
COMMITTED_LR = 0.03
COMMITTED_LEAVES = 15

# Registered list, in order (docs/prereg/A5_PERTURBATION.md). "lgbm" entries override LightGBM parameters,
# "cfg" entries override WalkForwardConfig fields, "noise" is the noise fraction of the inputs' sd.
PERTURBATIONS: dict[str, dict] = {
    "baseline": dict(label="a", desc="committed settings; must reproduce the committed forecasts"),
    "lr_half": dict(label="b", desc="learning_rate 0.015", lgbm=dict(learning_rate=COMMITTED_LR / 2)),
    "lr_double": dict(label="c", desc="learning_rate 0.06", lgbm=dict(learning_rate=COMMITTED_LR * 2)),
    "leaves_7": dict(label="d", desc="num_leaves 7", lgbm=dict(num_leaves=7)),
    "leaves_31": dict(label="d", desc="num_leaves 31", lgbm=dict(num_leaves=31)),
    "min_leaf_half": dict(label="e", desc="min_data_in_leaf halved", min_leaf_factor=0.5),
    "min_leaf_double": dict(label="e", desc="min_data_in_leaf doubled", min_leaf_factor=2.0),
    "noise_10": dict(label="f", desc="Gaussian noise, 10% of each continuous input's training sd", noise=0.10),
    "noise_25": dict(label="f", desc="Gaussian noise, 25% of each continuous input's training sd", noise=0.25),
    "block_42": dict(label="g", desc="refit block 42 days", cfg=dict(block=42)),
    "min_train_250": dict(label="h", desc="first training window 250 days", cfg=dict(min_train=250)),
    "single_seed": dict(label="i", desc="one LightGBM seed instead of the 5-seed average", cfg=dict(n_seeds=1)),
}


# ---------------------------------------------------------------------------
# Noise injection (tested in tests/test_a5_perturbation.py)
# ---------------------------------------------------------------------------


def noise_columns(ds: pd.DataFrame, features=BASE_FEATURES) -> list[str]:
    """The continuous inputs of the hybrid that live in the dataset: float columns of ``features``.
    Flags (bool) and integer codes (tdte, regime, weekday) are left out, and so are the four inputs the
    walk-forward computes itself (log_mcs, log_s_window, log_mcs_iv, log_implied_window)."""
    return [c for c in features if c in ds.columns and ds[c].dtype.kind == "f"]


def add_input_noise(ds: pd.DataFrame, columns, frac: float, seed: int, train_mask) -> pd.DataFrame:
    """Copy of ``ds`` with N(0, (frac * sd_c)^2) noise added to each column c in ``columns``.

    sd_c is the sample standard deviation of column c over the rows where ``train_mask`` is True (NaN
    ignored), so the noise scale uses only the first training window. One standard normal draw per row and
    column comes from ``numpy.random.default_rng(seed)``, in the order of ``columns``; the draws do not
    depend on ``frac``, so two fractions with the same seed use the same draw scaled. NaN cells stay NaN.
    A column whose training sd is zero or undefined is left unchanged (its draw is still consumed, so the
    other columns' draws do not shift).
    """
    rng = np.random.default_rng(seed)
    out = ds.copy()
    mask = np.asarray(train_mask, bool)
    for c in columns:
        x = out[c].to_numpy(float)
        z = rng.standard_normal(len(x))
        xt = x[mask]
        xt = xt[np.isfinite(xt)]
        sd = float(np.std(xt, ddof=1)) if xt.size > 1 else float("nan")
        if not np.isfinite(sd) or sd <= 0 or frac == 0:
            continue
        out[c] = x + frac * sd * z
    return out


# ---------------------------------------------------------------------------
# Patches: LightGBM parameter overrides and the mcsGARCH cache (both restored after each run)
# ---------------------------------------------------------------------------


def lgbm_override(overrides: dict):
    """Return a replacement for forecast._lgbm_params that applies ``overrides`` last."""
    orig = fc._lgbm_params

    def patched(cfg, seed, **g):
        p = orig(cfg, seed, **g)
        p.update(overrides)
        return p

    patched._orig = orig
    return patched


_EGARCH_CACHE: dict = {}
_MCS_CACHE: dict = {}
_orig_egarch = fc.daily_egarch_variance


def _h(a) -> str:
    return hashlib.sha1(np.ascontiguousarray(np.asarray(a, float)).tobytes()).hexdigest()


def cached_egarch(daily, train_size=None, **kw):
    key = (_h(daily), None if train_size is None else int(train_size), tuple(sorted(kw.items())))
    if key not in _EGARCH_CACHE:
        _EGARCH_CACHE[key] = _orig_egarch(daily, train_size=train_size, **kw)
    h, info = _EGARCH_CACHE[key]
    return np.array(h, copy=True), dict(info)


class CachedMcsGarch(McsGarch):
    """McsGarch whose fit is looked up by (settings, inputs) so identical fits are done once per process."""

    def fit(self, R, h, expiry_mask=None, compute_tstats: bool = False):
        R = np.asarray(R, float)
        h = np.asarray(h, float)
        em = np.zeros(R.shape[0], bool) if expiry_mask is None else np.asarray(expiry_mask, bool)
        key = (self.mean, self.dist, self.expiry_diurnal, self.smooth, R.shape, _h(R), _h(h), _h(em), compute_tstats)
        st = _MCS_CACHE.get(key)
        if st is None:
            McsGarch.fit(self, R, h, em, compute_tstats)
            _MCS_CACHE[key] = copy.deepcopy({k: v for k, v in self.__dict__.items()})
        else:
            self.__dict__.update(copy.deepcopy(st))
        return self


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_data(a, target: str):
    """Market, dataset and GEX sets, exactly as P99 stage 1 (scripts/run_pipeline.py loader)."""
    t0 = time.time()
    if a.source == "synthetic":
        ns = argparse.Namespace(source="synthetic", days=a.days, kappa=0.5, seed=a.seed, participant_noise=0.1)
        m = rp.load_market(ns)
        settings_note = "SYNTHETIC market (simulate_market)"
    else:
        ps = sb.PIPELINE_SETTINGS
        ns = argparse.Namespace(source="nse", raw=a.raw, symbol="NIFTY", oi_unit=ps["oi_unit"], lot_size=None,
                                rate=ps["rate"], start=ps["start"], end=a.end, min_coverage=ps["min_coverage"])
        m = rp.load_market(ns)
        info = json.load(open(HEADLINE_RUN / "run_info.json"))
        last = str(pd.Timestamp(m["days"][-1]).date())
        bad = sb.check_headline_settings(info, last)
        if len(m["days"]) != info["days"][2]:
            bad.append(f"days {len(m['days'])} vs headline {info['days'][2]}")
        settings_note = "settings identical to the headline run" if not bad else f"DIFFERS from headline: {bad}"
        print(f"[REAL DATA] {settings_note}", flush=True)
    print(f"market loaded: {len(m['days'])} days, {m['days'][0]} to {m['days'][-1]} ({time.time() - t0:.0f}s)", flush=True)
    panel = build_gex_panel(m)
    implied = build_implied_panel(m)
    sets = {k: v for k, v in GEX_SETS.items() if k != "oracle" or "dealer_next" in m}
    ds = build_dataset(m, panel, implied=implied, **rp.TARGETS[target])
    print(f"dataset {target} built: {len(ds)} rows ({time.time() - t0:.0f}s)", flush=True)
    return m, ds, sets, settings_note


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------


def run_settings(name: str, ds: pd.DataFrame) -> dict:
    """The concrete parameter values of one perturbation (for the summary row)."""
    p = PERTURBATIONS[name]
    base_min_leaf = 100 if ds["minute"].nunique() > 3 else 15     # as forecast.run_walkforward
    lg = dict(p.get("lgbm", {}))
    if "min_leaf_factor" in p:
        lg["min_data_in_leaf"] = int(round(base_min_leaf * p["min_leaf_factor"]))
    cfg = dict(p.get("cfg", {}))
    return dict(name=name, label=p["label"], desc=p["desc"],
                learning_rate=lg.get("learning_rate", COMMITTED_LR), num_leaves=lg.get("num_leaves", COMMITTED_LEAVES),
                min_data_in_leaf=lg.get("min_data_in_leaf", base_min_leaf), noise_frac=p.get("noise", 0.0),
                noise_seed=NOISE_SEED if "noise" in p else None,
                block=cfg.get("block", sb.PIPELINE_SETTINGS["block"]),
                min_train=cfg.get("min_train", sb.PIPELINE_SETTINGS["min_train"]),
                n_seeds=cfg.get("n_seeds", sb.PIPELINE_SETTINGS["n_seeds"]), seed=HEADLINE_SEED, lgbm=lg, cfg=cfg)


def run_one(name: str, target: str, m: dict, ds: pd.DataFrame, sets: dict, verbose: bool = False) -> tuple[pd.DataFrame, dict]:
    """Walk-forward of mcsgarch and hybrid_base under one perturbation. Returns (forecasts, settings)."""
    s = run_settings(name, ds)
    data = ds
    if s["noise_frac"] > 0:
        cols = noise_columns(ds)
        data = add_input_noise(ds, cols, s["noise_frac"], NOISE_SEED, ds["t"].to_numpy() < s["min_train"])
        s["noise_columns"] = cols
    cfg = sb.make_config(HEADLINE_SEED, rp.EVAL_MINUTES.get(target), sets, **s["cfg"])
    saved = (fc._lgbm_params, fc.daily_egarch_variance, fc.McsGarch)
    fc._lgbm_params = lgbm_override(s["lgbm"]) if s["lgbm"] else saved[0]
    fc.daily_egarch_variance, fc.McsGarch = cached_egarch, CachedMcsGarch
    try:
        pred, _ = fc.run_walkforward(m, data, cfg, verbose=verbose)
    finally:
        fc._lgbm_params, fc.daily_egarch_variance, fc.McsGarch = saved
    keep = sb.ID_COLS + MODELS
    return pred[keep].sort_values(["t", "minute"]).reset_index(drop=True), s


def evaluate(pred: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Mean QLIKE over anchors (as the scoreboard), gain in percent and the one-sided DM test on daily losses."""
    L = daily_losses(pred, MODELS, "qlike")
    dm = diebold_mariano(L["mcsgarch"], L["hybrid_base"], alternative="greater")
    qh, qm = float(np.mean(qlike(pred["y"], pred["hybrid_base"]))), float(np.mean(qlike(pred["y"], pred["mcsgarch"])))
    daily = pd.DataFrame({"date": L.index, "qlike_hybrid": L["hybrid_base"].to_numpy(), "qlike_mcsgarch": L["mcsgarch"].to_numpy()})
    return dict(n_days=int(dm["n"]), n_anchors=int(len(pred)), qlike_hybrid=qh, qlike_mcsgarch=qm,
                gain_pct=100 * (1 - qh / qm), dm_stat=dm["stat"], dm_p_one_sided=dm["p"]), daily


def reproduction(pred: pd.DataFrame, target: str) -> dict:
    """Forecast-by-forecast comparison with the committed v2 run (P99's check, seed 7)."""
    committed = HEADLINE_RUN / f"forecasts_{target}.csv"
    if not committed.exists():
        return {}
    wide = pred[sb.ID_COLS + ["mcsgarch"]].copy()
    wide[f"hyb_{HEADLINE_SEED}"] = pred["hybrid_base"].to_numpy()
    comm = sb.read_csv_exact(committed)
    for f in (wide, comm):   # dates as "YYYY-MM-DD" strings on both sides (the committed file is a CSV)
        f["date"] = pd.to_datetime(f["date"]).dt.strftime("%Y-%m-%d")
    r = sb.reproduction_check(wide, comm, target)
    return {f"repro_{k}": v for k, v in r.items() if k != "target"}


def raw_hashes(raw: Path) -> dict:
    out = {}
    for f in ("nifty_1min.csv", "india_vix.csv", "index_daily.csv"):
        p = raw / f
        if p.exists():
            out[f] = "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()
    for d in ("bhavcopy", "participant_oi", "index_close"):
        if (raw / d).is_dir():
            out[d + "_files"] = len(list((raw / d).iterdir()))
    return out


SUMMARY_COLS = ["source", "target", "name", "label", "desc", "learning_rate", "num_leaves", "min_data_in_leaf", "noise_frac",
                "noise_seed", "block", "min_train", "n_seeds", "seed", "n_days", "n_anchors", "qlike_hybrid", "qlike_mcsgarch",
                "gain_pct", "dm_stat", "dm_p_one_sided", "pass_positive_p05", "seconds", "from_work_file",
                "repro_hybrid_base_max_abs_rel_diff", "repro_mcsgarch_max_abs_rel_diff", "repro_y_max_abs_rel_diff",
                "repro_hybrid_base_n_identical", "repro_rows_matched"]


def append_summary(out: Path, row: dict):
    path = out / "summary.csv"
    df = pd.DataFrame([{c: row.get(c) for c in SUMMARY_COLS}])
    if path.exists():
        old = pd.read_csv(path)
        old = old[~((old["target"] == row["target"]) & (old["name"] == row["name"]) & (old["source"] == row["source"]))]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(path, index=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="next30", choices=list(rp.TARGETS))
    ap.add_argument("--perturbation", nargs="+", default=["all"], help="names from the registered list, or all")
    ap.add_argument("--out", default="results/a5_perturbation")
    ap.add_argument("--work", default=str(ROOT / "data" / "processed" / "a5_perturbation"))
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--end", default=sb.END)
    ap.add_argument("--source", default="nse", choices=["nse", "synthetic"])
    ap.add_argument("--days", type=int, default=120, help="synthetic only")
    ap.add_argument("--seed", type=int, default=1, help="synthetic market seed only (LightGBM base seed is 7)")
    ap.add_argument("--stop-after", type=float, default=None, help="minutes of wall clock after which no new run starts")
    ap.add_argument("--force", action="store_true", help="rerun even if the work file exists")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    if a.list:
        for k, v in PERTURBATIONS.items():
            print(f"{v['label']}  {k:16s} {v['desc']}")
        return
    names = list(PERTURBATIONS) if a.perturbation == ["all"] else a.perturbation
    unknown = [n for n in names if n not in PERTURBATIONS]
    if unknown:
        raise SystemExit(f"unknown perturbation(s) {unknown}; choose from {list(PERTURBATIONS)}")
    out, work = Path(a.out), Path(a.work)
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    tag = "REAL DATA" if a.source == "nse" else "SYNTHETIC"
    t_start = time.time()
    print(f"[{tag}] A5 perturbation study, target {a.target}, runs {names}", flush=True)
    m, ds, sets, settings_note = load_data(a, a.target)
    info_path = out / "run_info.json"
    info = json.load(open(info_path)) if info_path.exists() else {}
    info.setdefault("written_by", "AI agent (Agent C, backtest and stress pass), experiments/a5_perturbation.py")
    info.setdefault("plan", "docs/prereg/A5_PERTURBATION.md")
    info["source"] = tag
    info.setdefault("commands", [])
    info["commands"].append("PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/a5_perturbation.py " + " ".join(sys.argv[1:]))
    info.setdefault("runs", {})
    info["settings_check"] = settings_note
    info["days"] = [str(pd.Timestamp(m["days"][0]).date()), str(pd.Timestamp(m["days"][-1]).date()), len(m["days"])]
    if a.source == "nse":
        info["raw_hashes"] = raw_hashes(Path(a.raw))
    info["noise_columns"] = noise_columns(ds)
    try:
        import lightgbm
        info["versions"] = {"numpy": np.__version__, "pandas": pd.__version__, "lightgbm": lightgbm.__version__}
    except Exception:  # pragma: no cover
        pass
    for name in names:
        if a.stop_after is not None and (time.time() - t_start) / 60 > a.stop_after:
            print(f"stop-after {a.stop_after} min reached before {name}: NOT RUN", flush=True)
            info["runs"][f"{a.target}/{name}"] = {"status": "NOT RUN (time budget)"}
            break
        wf = work / f"forecasts_{a.target}_{name}.csv"
        t0 = time.time()
        if wf.exists() and not a.force:
            pred, s = sb.read_csv_exact(wf), run_settings(name, ds)
            from_work = True
        else:
            pred, s = run_one(name, a.target, m, ds, sets, verbose=a.verbose)
            tmp = wf.with_suffix(".tmp")
            pred.to_csv(tmp, index=False)
            os.replace(tmp, wf)
            from_work = False
        secs = round(time.time() - t0, 1)
        ev, daily = evaluate(pred)
        rep = reproduction(pred, a.target) if (name == "baseline" and a.source == "nse") else {}
        daily.to_csv(out / f"daily_losses_{a.target}_{name}.csv", index=False)
        row = {**{k: v for k, v in s.items() if k not in ("lgbm", "cfg", "noise_columns")}, **ev, **rep,
               "source": tag, "target": a.target, "seconds": secs, "from_work_file": from_work,
               "pass_positive_p05": bool(ev["gain_pct"] > 0 and ev["dm_p_one_sided"] < 0.05)}
        append_summary(out, row)
        info["runs"][f"{a.target}/{name}"] = {k: (None if (isinstance(v, float) and not np.isfinite(v)) else v)
                                            for k, v in row.items()}
        json.dump(info, open(info_path, "w"), indent=2, default=str)
        flag = "" if not rep else (" REPRODUCED EXACTLY" if rep.get("repro_hybrid_base_max_abs_rel_diff") == 0 else
                                   f" REPRODUCTION DIFFERS: max rel diff hybrid {rep.get('repro_hybrid_base_max_abs_rel_diff'):.3g}, "
                                   f"mcsgarch {rep.get('repro_mcsgarch_max_abs_rel_diff'):.3g}")
        print(f"  {name:16s} days {ev['n_days']:4d} QLIKE hybrid {ev['qlike_hybrid']:.6f} mcsgarch {ev['qlike_mcsgarch']:.6f} "
              f"gain {ev['gain_pct']:+.3f}% DM {ev['dm_stat']:.2f} p {ev['dm_p_one_sided']:.2e} "
              f"{'PASS' if row['pass_positive_p05'] else 'FAIL'} {secs:.0f}s{flag} (elapsed {(time.time() - t_start) / 60:.1f} min)", flush=True)
    info["seconds_total"] = round(time.time() - t_start, 1)
    json.dump(info, open(info_path, "w"), indent=2, default=str)
    print(f"done in {(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
