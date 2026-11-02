"""Second-person check of the paper's printed values typed into replicate_paper.py.

The comparison targets in ``experiments/replicate_paper.py`` (``P_T1``, ``P_T1_CORR``,
``P_T2``, ``P_T34``, ``P_T5``, ``P_T5_TEXT_PCT`` and ``P_T6``) were typed in by hand from
the paper's tables. A second person re-keyed the same cells BLIND (without looking at the
dicts) into ``results/replication_nse/rekey_blind.csv``, with table, page, row and column
labels exactly as printed. This script matches the two, cell by cell, and writes
``results/replication_nse/rekey_check.csv`` plus a short summary on stdout.

It only imports the constants from replicate_paper.py; it does not run the replication.

Usage:

    python experiments/rekey_check.py
    python experiments/rekey_check.py --blind my.csv --out check.csv

Columns of the output: cell (dict name, key and field), table, page, row_label,
column_label, printed_value (the blind string), blind_value (parsed), dict_value,
match (yes or no), abs_diff, status (compared, dict cell without blind entry, blind
entry without dict cell), dict_value_first_pass and resolution (from ``RESOLUTIONS``
below, which records every first-pass mismatch and how it was settled).
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BLIND = ROOT / "results" / "replication_nse" / "rekey_blind.csv"
OUT = ROOT / "results" / "replication_nse" / "rekey_check.csv"
DICT_NAMES = ("P_T1", "P_T1_CORR", "P_T2", "P_T34", "P_T5", "P_T5_TEXT_PCT", "P_T6")

# Order of the six numbers in each P_T5 and P_T6 tuple (see printed_checks() in replicate_paper.py).
T56_FIELDS = ("n_before", "median_before", "n_after", "median_after", "w", "p")
IDX = {"Nifty 50": "nifty", "Nifty Bank": "banknifty", "Bank Nifty": "banknifty"}
T1_ROWS = {"Mean": "mean", "StDev": "sd", "Minimum": "min", "Maximum": "max", "Skewness": "skew",
           "Kurtosis": "excess_kurtosis", "N": "n", "Anderson-Darling": "anderson_darling", "ADF": "adf"}
T2_ROWS = {"AR(1)": "phi", "MA(1)": "theta", "omega": "omega", "alpha": "alpha", "beta": "beta",
           "Shape": "shape", "LL": "loglik", "AIC": "aic"}
PANEL = {"Panel I": "nifty", "Panel II": "bn"}
POS = {"Nifty 50": 0, "Bank Nifty": 1}
TEXT_DAY = {"all days": "all", "Wednesday": "wed", "Thursday": "thu"}

# Every first-pass mismatch, keyed by cell, with the dict value seen on the first run and how it
# was settled after a second, careful look at the PDF (page images at 500 dpi and the PDF's text layer).
RESOLUTIONS: dict[str, dict] = {
    "P_T1[after,banknifty].anderson_darling": {
        "first_pass_dict_value": 466.153,
        "resolution": (
            "dict kept (466.153). Table 1, page 5, After / Nifty Bank Return, Anderson-Darling row: the PDF really "
            "prints 466,153*** with a comma character, while the other three cells of the row use a decimal point "
            "(649.191, 790.061, 492.399). Read with the comma as a thousands separator the value would be 466153, "
            "which is impossible next to the others; our own statistic on the after sample is 469.5 "
            "(table1_descriptive.csv). So the comma is a typesetting slip in the paper and the dict's decimal "
            "reading is right. Not a transcription slip of ours; no change to replicate_paper.py."
        ),
    },
}


# ---------------------------------------------------------------------------
# Parsing and matching
# ---------------------------------------------------------------------------

_SIGN_WORDS = (("decrease", -1), ("reduc", -1), ("lower", -1), ("fell", -1), ("fall", -1),
               ("increase", 1), ("higher", 1), ("rose", 1), ("rise", 1))
_NUMBER = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?")


def parse_printed(s) -> float:
    """Turn a printed string into a number.

    Significance stars and spaces are dropped, a typographic minus counts as a minus sign, and
    commas are read as thousands separators (so "466,153" is 466153, as printed). For an in-text
    percentage such as "decreases by 13%", the words give the sign.
    """
    t = str(s).strip().replace("−", "-").replace("*", "")
    t = re.sub(r"-\s+(?=\d)", "-", t)
    nums = _NUMBER.findall(t)
    if len(nums) != 1:
        raise ValueError(f"expected exactly one number in {s!r}, found {nums}")
    x = float(nums[0].replace(",", ""))
    if "%" in t:
        low = t.lower()
        for word, sign in _SIGN_WORDS:
            if word in low:
                x = sign * abs(x)
                break
    return x


def values_match(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=0.0)


def _cell(name: str, key, field: str | None = None) -> str:
    k = key if isinstance(key, tuple) else (key,)
    return f"{name}[{','.join(k)}]" + (f".{field}" if field else "")


def flatten(dicts: dict) -> dict[str, float]:
    """One entry per printed number in the dicts, keyed by a readable cell name."""
    out = {}
    for name, d in dicts.items():
        for key, v in d.items():
            if isinstance(v, dict):
                for f, x in v.items():
                    out[_cell(name, key, f)] = x
            elif isinstance(v, (tuple, list)) and len(v) == 2 and isinstance(v[0], (tuple, list)):
                m, tot = v  # connectedness block: 2x2 shares and the total
                for i in range(2):
                    for j in range(2):
                        out[_cell(name, key, f"share[{i}][{j}]")] = m[i][j]
                out[_cell(name, key, "total")] = tot
            elif isinstance(v, (tuple, list)):
                for f, x in zip(T56_FIELDS, v):
                    out[_cell(name, key, f)] = x
            else:
                out[_cell(name, key)] = v
    return out


def _sub_from_heading(h: str) -> str:
    h = h.lower()
    for start, sub in (("weekly and monthly", "all_expiry"), ("monthly expiry", "monthly"),
                       ("1st trading hour", "monthly_first_hour"), ("mid", "monthly_mid_hour"),
                       ("last trading hour", "monthly_last_hour")):
        if h.startswith(start):
            return sub
    raise KeyError(h)


def _day_from_heading(h: str) -> str:
    h = h.lower()
    return "wed" if "wednesday" in h else "thu" if "thursday" in h else "all"


def cell_for(row) -> str | None:
    """Map one blind row (table, row_label, column_label as printed) to a dict cell name."""
    t, r, c = row["table"], row["row_label"], row["column_label"]
    if t == "Table 1":
        if r == "Correlation":
            return _cell("P_T1_CORR", c.lower())
        side, idx = c.split(": ")
        return _cell("P_T1", (side.lower(), IDX[idx.replace(" Return", "")]), T1_ROWS[r])
    if t == "Table 2":
        side, idx = c.split(": ")
        side = "before" if side.lower().startswith("pre") else "after"
        return _cell("P_T2", (side, IDX[idx]), T2_ROWS[r])
    if t in ("Table 3", "Table 4"):
        heading, rr = r.rsplit(" | ", 1)
        panel, cc = c.split(": ") if ": " in c else ("", c)
        side, fam = ("before", "union") if t == "Table 3" else ("after", PANEL[panel])
        key = (side, fam, _sub_from_heading(heading))
        if rr == "To" and cc == "From":
            return _cell("P_T34", key, "total")
        if rr in POS and cc in POS:
            return _cell("P_T34", key, f"share[{POS[rr]}][{POS[cc]}]")
        return None
    if t in ("Table 5", "Table 6"):
        heading, rr = r.rsplit(" | ", 1)
        idx, cc = c.split(": ")
        if cc in ("Observation", "Median"):
            field = ("n_" if cc == "Observation" else "median_") + rr.lower()
        else:
            field = {"w-statistics": "w", "p-value": "p"}[cc]
        return _cell("P_T5" if t == "Table 5" else "P_T6", (_day_from_heading(heading), IDX[idx]), field)
    if t.startswith("Section 4.3 text"):
        idx, day = r.split(", ")
        return _cell("P_T5_TEXT_PCT", (TEXT_DAY[day], IDX[idx]))
    return None


def compare(blind: pd.DataFrame, dicts: dict, resolutions: dict | None = None) -> pd.DataFrame:
    """One row per blind entry and per dict cell, with both values and whether they match."""
    resolutions = resolutions or {}
    flat = flatten(dicts)
    rows, seen = [], set()
    for _, b in blind.iterrows():
        try:
            cell = cell_for(b)
        except (KeyError, ValueError):
            cell = None
        bv = parse_printed(b["printed_value"])
        rec = {"cell": cell or "", "table": b["table"], "page": b["page"], "row_label": b["row_label"],
               "column_label": b["column_label"], "printed_value": b["printed_value"], "blind_value": bv}
        if cell is None or cell not in flat:
            rec.update(dict_value=float("nan"), match="no", abs_diff=float("nan"), status="blind entry without dict cell")
        else:
            if cell in seen:
                raise ValueError(f"two blind rows map to the same cell {cell}")
            seen.add(cell)
            dv = float(flat[cell])
            rec.update(dict_value=dv, match="yes" if values_match(bv, dv) else "no", abs_diff=abs(bv - dv), status="compared")
        rows.append(rec)
    for cell, dv in flat.items():
        if cell not in seen:
            rows.append({"cell": cell, "table": "", "page": "", "row_label": "", "column_label": "", "printed_value": "",
                         "blind_value": float("nan"), "dict_value": float(dv), "match": "no", "abs_diff": float("nan"),
                         "status": "dict cell without blind entry"})
    out = pd.DataFrame(rows)
    out["dict_value_first_pass"] = [resolutions.get(c, {}).get("first_pass_dict_value", "") for c in out["cell"]]
    out["resolution"] = [resolutions.get(c, {}).get("resolution", "") for c in out["cell"]]
    return out


def summary(res: pd.DataFrame, resolutions: dict | None = None) -> str:
    resolutions = resolutions or {}
    comp = res[res["status"] == "compared"]
    bad = comp[comp["match"] == "no"]
    unresolved = bad[~bad["cell"].isin(list(resolutions))]
    lines = [f"Blind entries: {int((res['status'] != 'dict cell without blind entry').sum())}; "
             f"dict cells: {int((res['status'] != 'blind entry without dict cell').sum())}; compared: {len(comp)}",
             f"Match now: {int((comp['match'] == 'yes').sum())}; mismatch now: {len(bad)} "
             f"({len(bad) - len(unresolved)} explained in RESOLUTIONS, {len(unresolved)} unresolved)",
             f"First-pass mismatches recorded in RESOLUTIONS (fixed or explained): {len(resolutions)}"]
    for _, r in bad.iterrows():
        tag = "RESOLVED" if r["cell"] in resolutions else "UNRESOLVED"
        lines.append(f"  {tag} MISMATCH {r['cell']}: printed {r['printed_value']!r} ({r['table']}, p. {r['page']}) vs dict {r['dict_value']!r}")
    for cell, rec in resolutions.items():
        lines.append(f"  {cell}: {rec.get('resolution', '')[:110]}...")
    for status in ("dict cell without blind entry", "blind entry without dict cell"):
        un = res[res["status"] == status]
        lines.append(f"{status}: {len(un)}")
        for _, r in un.iterrows():
            lines.append(f"  {r['cell'] or (r['table'] + ' | ' + r['row_label'] + ' | ' + r['column_label'])}")
    return "\n".join(lines)


def load_dicts(path: Path = ROOT / "experiments" / "replicate_paper.py") -> dict:
    """Import replicate_paper.py as a module (its main() is not run) and return the P_T dicts."""
    spec = importlib.util.spec_from_file_location("_replicate_paper_constants", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {n: getattr(mod, n) for n in DICT_NAMES}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--blind", type=Path, default=BLIND)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    blind = pd.read_csv(a.blind, dtype=str, keep_default_na=False)
    res = compare(blind, load_dicts(), RESOLUTIONS)
    res.to_csv(a.out, index=False)
    print(summary(res, RESOLUTIONS))
    print(f"Wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
