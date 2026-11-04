"""Fast unit tests for experiments/referee_f2_positions.py (item 8; synthetic data only, no real files)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

_EXP = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(_EXP))
_spec = importlib.util.spec_from_file_location("referee_f2_positions", _EXP / "referee_f2_positions.py")
rp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rp)
rf = rp.rf

_HEADER = ("Client Type,Future Index Long,Future Index Short,Future Stock Long,Future Stock Short\t,"
           "Option Index Call Long,Option Index Put Long,Option Index Call Short,Option Index Put Short,"
           "Option Stock Call Long,Option Stock Put Long,Option Stock Call Short,Option Stock Put Short,"
           "Total Long Contracts\t,Total Short Contracts\n")


def _write_participant_file(path, rows):
    """rows: category -> (fut_long, fut_short, call_long, put_long, call_short, put_short)."""
    lines = ['""Participant wise Open Interest (no. of contracts) in Equity Derivatives as on Jan 01,2024"",,,,,,,,,,,,,,\n',
             _HEADER]
    tot = np.zeros(6)
    for cat in ("Client", "DII", "FII", "Pro"):
        v = np.array(rows[cat], float)
        tot += v
        fl, fs, cl, pl, cs, ps = v
        lines.append(f"{cat},{fl:.0f},{fs:.0f},0,0,{cl:.0f},{pl:.0f},{cs:.0f},{ps:.0f},0,0,0,0,0,0\n")
    fl, fs, cl, pl, cs, ps = tot
    lines.append(f"TOTAL,{fl:.0f},{fs:.0f},0,0,{cl:.0f},{pl:.0f},{cs:.0f},{ps:.0f},0,0,0,0,0,0\n")
    Path(path).write_text("".join(lines))


def test_raw_positions_reads_nse_files_and_matches_dealer_sign(tmp_path):
    rows = {"Client": (100, 160, 500, 400, 520, 450), "DII": (50, 40, 0, 50, 0, 0),
            "FII": (120, 60, 200, 300, 150, 200), "Pro": (30, 40, 300, 250, 330, 350)}
    _write_participant_file(tmp_path / "fao_participant_oi_02012024.csv", rows)
    rows2 = {k: tuple(x + 10 for x in v) for k, v in rows.items()}
    _write_participant_file(tmp_path / "fao_participant_oi_03012024.csv", rows2)
    pos = rp.raw_positions(tmp_path)
    assert list(pos.index) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    r = pos.loc["2024-01-02"]
    tot_fut, tot_call, tot_put = 300.0, 1000.0, 1000.0  # TOTAL long of futures, calls, puts
    assert np.isclose(r["pro_call"], (300 - 330) / tot_call)
    assert np.isclose(r["pro_put"], (250 - 350) / tot_put)
    assert np.isclose(r["pro_fut"], (30 - 40) / tot_fut)
    assert np.isclose(r["fii_fut"], (120 - 60) / tot_fut)
    assert np.isclose(r["pro_call_minus_put"], r["pro_call"] - r["pro_put"])
    assert np.isclose(r["pro_call_plus_put"], r["pro_call"] + r["pro_put"])
    nets = rp.dsg.participant_nets(tmp_path)
    for c in ("pro", "fii", "dii", "client"):
        for s in ("call", "put"):
            assert np.allclose(pos[f"{c}_{s}"], nets[f"{c}_{s}"])


def test_evening_values_are_lagged_and_forward_filled():
    idx = pd.bdate_range("2024-03-04", periods=5)
    gex = pd.DataFrame({"pro": [1.0, -2.0, 3.0, -4.0, 5.0], "fii": 1.0, "client": -1.0, "gross": 10.0}, index=idx)
    raw = pd.DataFrame({f"{c}_{p}": [1.0, 2.0, 4.0] for c in rp.CATS8 for p in rp.RAW_POS},
                       index=[idx[0], idx[1], idx[3]])  # no file on idx[2]
    ev = rp.evening_values(raw, gex, idx)
    assert np.isnan(ev["pro_call"].iloc[0])
    assert ev["pro_call"].iloc[1:].tolist() == [1.0, 2.0, 2.0, 4.0]  # previous evening; idx[2] forward-filled
    assert ev["pro_gex"].iloc[1:].tolist() == [0.1, -0.2, 0.3, -0.4]


def test_make_state_dummy_and_standardized_level():
    v = np.array([-2.0, -1.0, 0.0, 1.0, 5.0])
    S, dmy = rp.make_state(v, "dummy")
    assert S.tolist() == [1.0, 1.0, 0.0, 0.0, 0.0] and np.array_equal(S, dmy)
    L, dmy = rp.make_state(v, "level")
    assert np.isclose(L.mean(), 0) and np.isclose(L.std(), 1)
    assert dmy.tolist() == [1.0, 1.0, 0.0, 0.0, 0.0]
    assert np.all(np.diff(L) > 0)  # order kept


def test_joint_episodes_start_when_any_state_changes():
    a = np.array([1, 1, 1, 0, 0, 0, 1])
    b = np.array([0, 0, 1, 1, 1, 0, 0])
    assert rp.joint_episodes([a, b]).tolist() == [0, 0, 1, 2, 2, 3, 4]
    assert rp.joint_episodes([a]).tolist() == rf.episode_ids(a).tolist()


def test_hac_wald_matches_statsmodels():
    rng = np.random.default_rng(0)
    n = 400
    X = np.column_stack([np.ones(n), rng.normal(size=(n, 3))])
    y = X @ np.array([0.2, 0.3, 0.0, -0.1]) + rng.normal(size=n)
    res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    J = [1, 3]
    V = res.cov_params()[np.ix_(J, J)]
    W_ref = res.params[J] @ np.linalg.solve(V, res.params[J])
    w = rp.hac_wald(y, X, J, lags=5)
    assert np.isclose(w["W"], W_ref, rtol=1e-8)
    assert w["q"] == 2


def _persistent_level(rng, n, phi=0.97):
    z = np.zeros(n)
    for t in range(1, n):
        z[t] = phi * z[t - 1] + rng.normal()
    return z


def test_wild_cluster_wald_statistic_size_and_power():
    rng = np.random.default_rng(1)
    n, G = 600, 40
    g = np.repeat(np.arange(G), n // G)
    r = rng.normal(scale=50, size=n)
    z1, z2 = _persistent_level(rng, n), _persistent_level(rng, n)
    X = np.column_stack([np.ones(n), r, z1, z1 * r, z2, z2 * r])
    y0 = -0.03 * r + rng.normal(scale=10, size=n)
    # the bootstrap's observed statistic is the CR1 Wald of statsmodels
    res = sm.OLS(y0, X).fit(cov_type="cluster", cov_kwds={"groups": g})
    J = [3, 5]
    W_ref = res.params[J] @ np.linalg.solve(res.cov_params()[np.ix_(J, J)], res.params[J])
    w0 = rp.wild_cluster_wald(y0, X, J, g, 199, np.random.default_rng(2))
    assert np.isclose(w0["W"], W_ref, rtol=1e-8) and w0["G"] == G
    assert w0["p"] > 0.01
    y1 = y0 + 0.02 * z1 * r
    w1 = rp.wild_cluster_wald(y1, X, J, g, 199, np.random.default_rng(3))
    assert w1["p"] < 0.02


def _synthetic(c_level=0.0, c_dummy=0.0, n=700, seed=5):
    rng = np.random.default_rng(seed)
    z = _persistent_level(rng, n)
    r = rng.normal(scale=60, size=n)
    dmy = (z < 0).astype(float)
    y = -0.03 * r + c_level * (z / z.std()) * r + c_dummy * dmy * r + rng.normal(scale=12, size=n)
    quarters = np.repeat(np.arange(n // 60 + 1), 60)[:n]
    return y, r, z, quarters


def test_infer_state_recovers_a_level_interaction_and_reports_every_method():
    y, r, z, q = _synthetic(c_level=0.04)
    S, dmy = rp.make_state(z, "level")
    rows = pd.DataFrame(rp.infer_state(y, r, S, dmy, q, {"n_wild": 99, "n_perm": 99}, [1], binary=False))
    assert set(rows["method"]) == {"newey_west_5", "wild_cluster_episode", "wild_cluster_quarter",
                                   "block_permutation_60_t", "block_permutation_60_coef"}
    assert abs(rows["coef"].iloc[0] - 0.04) < 0.015
    assert (rows["p"] <= 0.05).all()
    assert "99 valid draws" in rows.loc[rows["method"] == "block_permutation_60_coef", "detail"].iloc[0]
    assert rows["n_episodes"].iloc[0] == len(rf.runs_of(dmy))


def test_infer_state_dummy_null_and_rare_state():
    y, r, z, q = _synthetic(seed=7)
    S, dmy = rp.make_state(z, "dummy")
    rows = pd.DataFrame(rp.infer_state(y, r, S, dmy, q, {"n_wild": 99, "n_perm": 199}, [2], binary=True))
    assert (rows.loc[rows["method"].str.startswith("block_permutation"), "p"] > 0.01).all()
    rare = np.zeros(len(y))
    rare[:3] = 1.0
    out = rp.infer_state(y, r, rare, rare, q, {"n_wild": 9, "n_perm": 9}, [3], binary=True)
    assert out[0]["method"] == "not estimable"


def test_horse_race_picks_the_state_that_drives_the_outcome():
    y, r, z, q = _synthetic(c_dummy=-0.07, seed=9)
    rng = np.random.default_rng(10)
    a, _ = rp.make_state(z, "dummy")
    b = a.copy()
    flip = rng.random(len(a)) < 0.25  # a correlated but different state
    b[flip] = 1 - b[flip]
    lvl, lvl_d = rp.make_state(_persistent_level(rng, len(a)), "level")
    rows = pd.DataFrame(rp.infer_multi(y, r, {"a_dummy": a, "b_dummy": b, "x_level": lvl},
                                       {"a_dummy": a, "b_dummy": b, "x_level": lvl_d}, q,
                                       {"n_wild": 99}, [4], joint=["b_dummy", "x_level"]))
    nw = rows[rows["method"] == "newey_west_5"].set_index("state")
    assert abs(nw.loc["a_dummy", "coef"] - (-0.07)) < 0.03 and nw.loc["a_dummy", "p"] < 0.01
    assert nw.loc["b_dummy", "p"] > 0.01
    jt = rows[rows["state"] == "joint"]
    assert set(jt["method"]) == {"newey_west_5_wald", "wild_cluster_joint_episode_wald", "wild_cluster_quarter_wald"}
    assert (jt["p"] > 0.01).all()
