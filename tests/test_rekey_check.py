"""Fast tests for experiments/rekey_check.py on a tiny synthetic re-key CSV and dict (no PDF needed)."""

import csv
import importlib.util
import math
from pathlib import Path

import pandas as pd
import pytest

_PATH = Path(__file__).resolve().parents[1] / "experiments" / "rekey_check.py"
_spec = importlib.util.spec_from_file_location("rekey_check", _PATH)
rk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rk)

COLS = ["table", "page", "row_label", "column_label", "printed_value", "note"]


def _blind(tmp_path, rows):
    p = tmp_path / "blind.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        w.writerows(rows)
    return pd.read_csv(p, dtype=str, keep_default_na=False)  # read as main() does


def _dicts():
    return {
        "P_T1": {("before", "nifty"): dict(n=23039, mean=-0.000001)},
        "P_T1_CORR": {"before": 0.854},
        "P_T2": {("after", "banknifty"): dict(aic=-13.160)},
        "P_T34": {("after", "bn", "monthly_first_hour"): ([[62.44, 37.56], [34.41, 65.59]], 71.97)},
        "P_T5": {("thu", "banknifty"): (4680, 425000, 4680, 411000, 21478045.0, 0.0011)},
        "P_T5_TEXT_PCT": {("all", "banknifty"): -13},
    }


ROWS = [
    ["Table 1", "5", "Mean", "Before: Nifty 50 Return", "− 0.000001", ""],
    ["Table 1", "5", "N", "Before: Nifty 50 Return", "23,039", ""],
    ["Table 1", "5", "Correlation", "Before", "0.854***", ""],
    ["Table 2", "7", "AIC", "Post announcement: Nifty Bank", "-13.160", ""],
    ["Table 4", "11", "1st trading hour Monthly expiry of Bank Nifty | Nifty 50", "Panel II: Nifty 50", "62.44", ""],
    ["Table 4", "11", "1st trading hour Monthly expiry of Bank Nifty | Bank Nifty", "Panel II: Nifty 50", "34.41", ""],
    ["Table 4", "11", "1st trading hour Monthly expiry of Bank Nifty | To", "Panel II: From", "70.97", "total"],
    ["Table 5", "12", "Change in Volume on Thursday Before and after the change in expiration of Bank Nifty | Before",
     "Nifty Bank: w-statistics", "21,478,045", ""],
    ["Table 5", "12", "Change in Volume on Thursday Before and after the change in expiration of Bank Nifty | After",
     "Nifty Bank: Median", "411,000", ""],
    ["Table 5", "12", "Change in Volume on Thursday Before and after the change in expiration of Bank Nifty | Before",
     "Nifty Bank: p-value", "0.0011", ""],
    ["Section 4.3 text (about Table 5)", "12", "Bank Nifty, all days", "median volume % change", "decreases by 13%", ""],
    ["Table 6", "13", "Change in Volatility Before and after the change in expiration of Bank Nifty | Before",
     "Nifty 50: Median", "0.00024", "not in the synthetic dict"],
]


def test_parse_printed_handles_minus_stars_commas_and_words():
    assert rk.parse_printed("− 23.05000***") == -23.05
    assert rk.parse_printed("-0.054778**") == -0.054778
    assert rk.parse_printed("158,616.4") == 158616.4
    assert rk.parse_printed("2,174,645,383") == 2174645383
    assert rk.parse_printed("466,153***") == 466153  # a comma is read as a thousands separator, as printed
    assert rk.parse_printed("decreases by 13%") == -13
    assert rk.parse_printed("reduced by 3%") == -3
    assert rk.parse_printed("19% higher") == 19
    assert rk.parse_printed("increased by 32%") == 32
    with pytest.raises(ValueError):
        rk.parse_printed("not readable")


def test_compare_matches_flags_mismatch_and_lists_unmatched_both_ways(tmp_path):
    blind = _blind(tmp_path, ROWS)
    res = rk.compare(blind, _dicts(), {"P_T34[after,bn,monthly_first_hour].total": {
        "first_pass_dict_value": 71.97, "resolution": "synthetic: dict slip"}})
    by = res.set_index("cell")

    # matches, including a typographic minus, stars, thousands commas and a worded percentage
    for cell in ("P_T1[before,nifty].mean", "P_T1[before,nifty].n", "P_T1_CORR[before]", "P_T2[after,banknifty].aic",
                 "P_T34[after,bn,monthly_first_hour].share[0][0]", "P_T34[after,bn,monthly_first_hour].share[1][0]",
                 "P_T5[thu,banknifty].w", "P_T5[thu,banknifty].median_after", "P_T5[thu,banknifty].p",
                 "P_T5_TEXT_PCT[all,banknifty]"):
        assert by.loc[cell, "match"] == "yes", cell
        assert by.loc[cell, "abs_diff"] == 0

    # the planted mismatch: printed 70.97, dict 71.97
    tot = by.loc["P_T34[after,bn,monthly_first_hour].total"]
    assert tot["match"] == "no" and math.isclose(tot["abs_diff"], 1.0)
    assert tot["resolution"] == "synthetic: dict slip" and tot["dict_value_first_pass"] == 71.97

    # dict cells with no blind entry, and a blind entry with no dict cell, are both listed
    missing = set(res.loc[res["status"] == "dict cell without blind entry", "cell"])
    assert missing == {"P_T34[after,bn,monthly_first_hour].share[0][1]", "P_T34[after,bn,monthly_first_hour].share[1][1]",
                       "P_T5[thu,banknifty].n_before", "P_T5[thu,banknifty].median_before", "P_T5[thu,banknifty].n_after"}
    extra = res[res["status"] == "blind entry without dict cell"]
    assert len(extra) == 1 and extra.iloc[0]["table"] == "Table 6"
    assert (res.loc[res["status"] != "compared", "match"] == "no").all()

    text = rk.summary(res, {"P_T34[after,bn,monthly_first_hour].total": {"resolution": "x"}})
    assert "RESOLVED MISMATCH P_T34[after,bn,monthly_first_hour].total" in text
    assert "0 unresolved" in text


def test_unresolved_mismatch_is_reported(tmp_path):
    blind = _blind(tmp_path, ROWS[:1])
    res = rk.compare(blind, {"P_T1": {("before", "nifty"): dict(mean=-0.00001)}})
    assert res.iloc[0]["match"] == "no"
    assert "UNRESOLVED MISMATCH P_T1[before,nifty].mean" in rk.summary(res)


def test_two_blind_rows_for_one_cell_are_refused(tmp_path):
    blind = _blind(tmp_path, [ROWS[0], ROWS[0]])
    with pytest.raises(ValueError):
        rk.compare(blind, _dicts())
