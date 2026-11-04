"""End-to-end test of the real-data path on SYNTHETIC files written in NSE layouts."""

import numpy as np
import pandas as pd

from niftygex.data import build_market_from_nse, load_bhavcopy, load_participant_oi
from niftygex.features import build_dataset, build_gex_panel
from niftygex.simulate import SimConfig, export_nse_layout, simulate_market


def test_nse_layout_roundtrip(tmp_path):
    m = simulate_market(SimConfig(n_days=12, kappa=0.0, seed=4))
    paths = export_nse_layout(m, tmp_path)
    one = sorted(paths["bhavcopy"].glob("*.csv"))[0]
    bc = load_bhavcopy(one)
    assert len(bc) == len(m["chains"][0]) and set(bc.columns) >= {"strike", "is_call", "oi", "settle"}
    tbl = load_participant_oi(sorted(paths["participant"].glob("*.csv"))[0])
    assert {"Client", "DII", "FII", "Pro", "TOTAL"} <= set(tbl.index)
    assert np.isclose(tbl.loc["TOTAL", "option_index_call_long"], tbl.loc["TOTAL", "option_index_call_short"], rtol=1e-6)

    mk = build_market_from_nse(paths["bars"], paths["bhavcopy"], paths["participant"], paths["vix"], oi_unit="units")
    assert mk["R"].shape == m["R"].shape
    assert np.allclose(mk["R"], m["R"], atol=1e-9)
    assert np.allclose(mk["vix"], m["vix"], rtol=1e-6)
    # OI converted back from units to contracts (the loader sorts by strike, so compare in that order)
    key = ["is_call", "strike"]
    assert np.allclose(mk["chains"][3].sort_values(key)["oi"].to_numpy(), m["chains"][3].sort_values(key)["oi"].to_numpy())
    # participant weights keep their sign
    assert np.all(np.sign(mk["participant"]["pro_call"]) == np.sign(m["participant"]["pro_call"]))
    panel = build_gex_panel(mk)
    ds = build_dataset(mk, panel, horizon=30, step=60)
    assert len(ds) > 0 and np.isfinite(ds["y"]).all()
