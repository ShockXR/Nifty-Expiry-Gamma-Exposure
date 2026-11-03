"""Tests for scripts/trace_numbers.py (P98 number tracing) on small synthetic files in tmp_path.

Nothing here reads the repository's documents, results or the private working paper.
"""

import csv
import gzip
import importlib.util
import json
import sys
from decimal import Decimal
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "trace_numbers.py"
_spec = importlib.util.spec_from_file_location("trace_numbers", _PATH)
tn = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = tn  # dataclasses look the module up by name
_spec.loader.exec_module(tn)


@pytest.fixture(autouse=True)
def _fresh_cache():
    tn._CSV_CACHE.clear()
    yield
    tn._CSV_CACHE.clear()


def write_csv(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def rule(spec: str = ""):
    return tn.parse_rule(spec)


# ------------------------------------------------------------------------------------------
# Printed numbers and rounding
# ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text,value,place", [
    ("5.8", Decimal("5.8"), -1),
    ("-0.057", Decimal("-0.057"), -3),
    ("21,239", Decimal("21239"), 0),
    ("0.0006", Decimal("0.0006"), -4),
    ("99.6%", Decimal("99.6"), -1),
    ("<0.001", Decimal("0.001"), -3),
    ("+1.14", Decimal("1.14"), -2),
    ("4e-6", Decimal("4e-6"), -6),
    ("4.3e-6", Decimal("4.3e-6"), -7),
    ("10^-256", Decimal("1e-256"), -256),
    ("\u22120.89", Decimal("-0.89"), -2),  # a typographic minus sign (U+2212)
])
def test_parse_printed(text, value, place):
    v, p = tn.parse_printed(text)
    assert v == value
    assert p == place


def test_parse_printed_rejects_words():
    with pytest.raises(ValueError):
        tn.parse_printed("six")


@pytest.mark.parametrize("printed,file_value,ok", [
    ("5.8", 5.84, True),
    ("-0.057", -0.0574, True),
    ("5.8", 5.9, False),
    ("5.8", 5.86, False),
    ("5.8", 5.75, True),           # a tie at the half unit counts as a match
    ("5.8", 5.85, True),
    ("-0.057", -0.05747, True),
    ("-0.069", -0.06951, False),   # rounds to -0.070
    ("0.0006", 0.000664, False),   # rounds to 0.0007
    ("0.95", 0.9414, False),
    ("21,239", 21239.0, True),
    ("1,255", 1255.23, True),
    ("4e-6", 4.34e-6, True),
    ("84.00", 84.0, True),
    ("84.00", 84.006, False),
])
def test_round_mode(printed, file_value, ok):
    assert tn.compare(printed, file_value, rule())[0] is ok


def test_nan_file_value_is_not_a_match():
    assert tn.compare("5.8", float("nan"), rule())[0] is False


def test_percent_is_not_scaled_unless_the_registry_says_so():
    # A printed "54%" is compared with the file value as it stands ...
    assert tn.compare("54%", 0.5354, rule())[0] is False
    assert tn.compare("54", 0.5354, rule())[0] is False
    assert tn.compare("54%", 53.54, rule())[0] is True
    # ... and only an explicit x100 turns a fraction into a percent.
    r = rule("x100")
    assert tn.compare("54%", tn.apply_transforms(0.5354, r.transforms), r)[0] is True


@pytest.mark.parametrize("spec,printed,file_value,ok", [
    ("lt", "0.001", 0.0001, True),
    ("lt", "0.001", 0.001, False),
    ("le", "0.010", 0.010, True),
    ("gt", "0.05", 0.06, True),
    ("exact", "40", 40.0, True),
    ("exact", "40", 40.2, False),
    ("abs=0.5", "100", 100.4, True),
    ("abs=0.5", "100", 100.6, False),
    ("rel=0.05", "100", 102, True),
    ("rel=0.05", "100", 106, False),
    ("oom", "10^-256", 2.89e-256, True),
    ("oom", "10^-256", 2.89e-255, False),
    ("oom", "1e-12", 1.39e-12, True),
])
def test_other_modes(spec, printed, file_value, ok):
    assert tn.compare(printed, file_value, rule(spec))[0] is ok


def test_text_and_date_modes():
    assert tn.compare("abc", "abc", rule("str"))[0] is True
    assert tn.compare("abc", "abd", rule("str"))[0] is False
    assert tn.compare("14 November 2022", "2022-11-14", rule("date"))[0] is True
    assert tn.compare("14 November 2022", "2022-11-15", rule("date"))[0] is False
    assert tn.compare("10 Feb 2023", "2023-02-10 00:00:00", rule("date"))[0] is True


def test_rule_parsing():
    r = rule("max; neg ;mul=100;lt")
    assert (r.mode, r.combiner, r.transforms) == ("lt", "max", ["neg", "mul=100"])
    assert rule("").mode == "round"
    assert rule("abs=0.2").tol == 0.2
    with pytest.raises(ValueError):
        rule("percent")
    with pytest.raises(ValueError):
        rule("mul=abc")


@pytest.mark.parametrize("transforms,x,expected", [
    (["x100"], 0.5354, 53.54),
    (["neg"], -5.8, 5.8),
    (["abs"], -2.0, 2.0),
    (["one_minus", "x100"], 0.027778, 97.2222),
    (["ratio_pct"], 1.2113, 21.13),
    (["expm1_pct"], 0.0, 0.0),
    (["half"], 84.0, 42.0),
    (["neg", "mul=100"], -0.0294, 2.94),
    (["divby=60"], 5961.7, 99.3617),
])
def test_transforms(transforms, x, expected):
    assert tn.apply_transforms(x, transforms) == pytest.approx(expected, rel=1e-4)


def test_expm1_pct_value():
    assert tn.apply_transforms(0.191710, ["expm1_pct"]) == pytest.approx(21.1319, rel=1e-4)


@pytest.mark.parametrize("combiner,values,expected", [
    ("sum", [0.0107, -0.0682], -0.0575),
    ("diff", [190.0, 21.0], 169.0),
    ("div", [0.04, 0.2], 0.2),
    ("mul", [0.99894, 943.0], 942.0),
    ("max", [-5.8, -7.6, -6.4], -5.8),
    ("min", [-5.8, -7.6, -6.4], -7.6),
])
def test_combiners(combiner, values, expected):
    assert tn.combine(values, combiner) == pytest.approx(expected, rel=1e-3)


def test_combiner_errors():
    assert tn.combine([3.0], None) == 3.0
    with pytest.raises(ValueError):
        tn.combine([1.0, 2.0], None)
    with pytest.raises(ValueError):
        tn.combine([1.0, 2.0, 3.0], "diff")


# ------------------------------------------------------------------------------------------
# Selectors
# ------------------------------------------------------------------------------------------

@pytest.fixture
def table(tmp_path):
    return write_csv(tmp_path / "t.csv", [
        {"group": "all", "model": "hybrid", "n": "929", "pct": "-5.7988", "flag": "True", "label": "binary, all days"},
        {"group": "all", "model": "har", "n": "929", "pct": "6.1482", "flag": "False", "label": "x"},
        {"group": "expiry", "model": "hybrid", "n": "169", "pct": "-6.6426", "flag": "True", "label": "y"},
        {"group": "expiry", "model": "har", "n": "169", "pct": "", "flag": "False", "label": "z"},
    ])


def test_csv_selector_single_cell(table):
    assert tn.select_one(table, "csv:group=all&model=hybrid@pct") == "-5.7988"
    # values with spaces and commas, and numeric equality (929 == 929.0)
    assert tn.select_one(table, "csv:label=binary, all days@model") == "hybrid"
    assert tn.select_one(table, "csv:n=929.0&model=har@pct") == "6.1482"


def test_csv_selector_operators_and_aggregates(table):
    assert tn.select_one(table, "csv:@#count") == 4.0
    assert tn.select_one(table, "csv:flag=True@#count") == 2.0
    assert tn.select_one(table, "csv:n>200@#count") == 2.0
    assert tn.select_one(table, "csv:n<=169@#count") == 2.0
    assert tn.select_one(table, "csv:group!=all@#count") == 2.0
    assert tn.select_one(table, "csv:pct<0@#count") == 2.0  # the empty cell is not below 0
    assert tn.select_one(table, "csv:model=hybrid@#min:pct") == pytest.approx(-6.6426)
    assert tn.select_one(table, "csv:model=hybrid@#max:pct") == pytest.approx(-5.7988)
    assert tn.select_one(table, "csv:@#sum:flag") == 2.0  # True counts as 1
    assert tn.select_one(table, "csv:group=expiry@#mean:pct") == pytest.approx(-6.6426)  # empty cells skipped


def test_csv_topshare(tmp_path):
    f = write_csv(tmp_path / "trades.csv", [{"gross": str(v)} for v in (10, 5, 3, 1, 1)])
    assert tn.select_one(f, "csv:@#topshare:gross:2") == pytest.approx(15 / 20)


def test_csv_selector_errors(table, tmp_path):
    with pytest.raises(tn.SourceError):
        tn.select_one(table, "csv:group=all@pct")  # two rows
    with pytest.raises(tn.SourceError):
        tn.select_one(table, "csv:group=none@pct")  # no row
    with pytest.raises(tn.SourceError):
        tn.select_one(table, "csv:group=all&model=hybrid@missing")
    with pytest.raises(tn.SourceError):
        tn.select_one(table, "csv:nocol=1@pct")
    with pytest.raises(tn.SourceError):
        tn.select_one(tmp_path / "absent.csv", "csv:@#count")
    with pytest.raises(ValueError):
        tn.select_one(table, "csv:group=all")  # no '@column'
    with pytest.raises(ValueError):
        tn.select_one(table, "xml:whatever")


def test_gzip_csv(tmp_path):
    f = tmp_path / "x.csv.gz"
    with gzip.open(f, "wt", encoding="utf-8", newline="") as fh:
        fh.write("a,b\n1,2.5\n")
    assert tn.select_one(f, "csv:a=1@b") == "2.5"


def test_json_selector(tmp_path):
    f = tmp_path / "v.json"
    f.write_text(json.dumps({"mde": 4.9, "holm": {"S1": 0.0153}, "list": [1, {"x": 7}], "a.b": 3}))
    assert tn.select_one(f, "json:mde") == 4.9
    assert tn.select_one(f, "json:holm.S1") == 0.0153
    assert tn.select_one(f, "json:list[1].x") == 7
    assert tn.select_one(f, r"json:a\.b") == 3
    with pytest.raises(tn.SourceError):
        tn.select_one(f, "json:holm.S9")
    with pytest.raises(tn.SourceError):
        tn.select_one(f, "json:holm")  # a container, not a value


def test_regex_selector(tmp_path):
    f = tmp_path / "log.txt"
    f.write_text("run took 303 seconds\nslope = -0.0294 (p 0.015)\n")
    assert tn.select_one(f, r"regex:slope = (\S+)") == "-0.0294"
    assert tn.select_one(f, r"regex:\d+ seconds") == "303 seconds"
    with pytest.raises(tn.SourceError):
        tn.select_one(f, "regex:absent (\\d+)")


@pytest.fixture
def forecasts(tmp_path):
    # Two days of two anchors. Day 1: the model is exact, the benchmark is off; day 2: the reverse.
    rows = []
    for date, minute, y, model, bench in [
        ("2024-01-01", 0, 1.0, 1.0, 2.0), ("2024-01-01", 15, 1.0, 1.0, 2.0),
        ("2024-01-02", 0, 1.0, 3.0, 1.0), ("2024-01-02", 15, 1.0, 1.0, 1.0),
        ("2024-01-03", 0, 2.0, 2.0, 4.0), ("2024-01-03", 15, 2.0, 2.0, 2.0),
    ]:
        rows.append({"date": date, "minute": minute, "y": y, "m": model, "b": bench})
    return write_csv(tmp_path / "forecasts_next15.csv", rows)


def test_daily_win_share(forecasts):
    # The model's daily mean QLIKE is lower on day 1 and day 3, higher on day 2.
    assert tn.select_one(forecasts, "derived:daily_win_share?model=m&bench=b") == pytest.approx(2 / 3)


def test_top_k_sq_error_share(forecasts):
    # squared errors of b: 1, 1, 0, 0, 4, 0 -> the largest one is 4 of 6
    assert tn.select_one(forecasts, "derived:top_k_sq_error_share?model=b&k=1") == pytest.approx(4 / 6)


def test_minute_gain_share(forecasts):
    q = tn._qlike
    gains = {0: [q(1, 2) - 0, 0 - q(1, 3), q(2, 4) - 0], 15: [q(1, 2) - 0, 0.0, 0.0]}
    total = sum(gains[0]) + sum(gains[15])
    got = tn.select_one(forecasts, "derived:minute_gain_share?model=m&bench=b&minute=0")
    assert got == pytest.approx(sum(gains[0]) / total)


def test_derived_errors(forecasts):
    with pytest.raises(ValueError):
        tn.select_one(forecasts, "derived:nothing")
    with pytest.raises(tn.SourceError):
        tn.select_one(forecasts, "derived:daily_win_share?model=zz&bench=b")


def test_resolve_value_multi_and_transforms(tmp_path, table):
    raw, v = tn.resolve_value("t.csv || t.csv", "csv:group=all&model=hybrid@pct || csv:group=expiry&model=hybrid@pct",
                              rule("min;neg"), tmp_path)
    assert v == pytest.approx(6.6426)
    assert "|" in raw
    with pytest.raises(ValueError):
        tn.resolve_value("t.csv || t.csv || t.csv", "csv:@#count || csv:@#count", rule("sum"), tmp_path)
    with pytest.raises(tn.SourceError):
        tn.resolve_value("t.csv", "csv:group=all&model=hybrid@label", rule(), tmp_path)  # not a number


# ------------------------------------------------------------------------------------------
# Documents and snippets
# ------------------------------------------------------------------------------------------

def test_snippet_search_ignores_line_breaks():
    d = tn.Document(Path("x.md"), "the last hour 5.8%,\n7.6% and   6.4% better")
    assert d.find_all("hour 5.8%, 7.6% and 6.4%") == [d.norm.find("hour")]
    assert d.find_all("not there") == []


@pytest.mark.parametrize("snippet,value,found", [
    ("by 5.8% and", "5.8", True),
    ("by 15.8% and", "5.8", False),      # inside a longer number
    ("by 5.81 and", "5.8", False),
    ("slope -0.057 against", "-0.057", True),
    ("+1% to +22%", "1", True),
    ("+11% to +22%", "1", False),
    ("(+1.14 log points", "1.14", True),  # a printed "+" is optional
    ("as small as 10^-256;", "10^-256", True),
    ("from 14 November 2022 to", "14 November 2022", True),
])
def test_value_in_snippet(snippet, value, found):
    assert (tn.value_offset_in_snippet(snippet, value) is not None) is found


def test_scan_numbers_masks_non_results():
    text = ("On 3 August 2026 at 15:29, in 2025 and November 2024 (Table 4, Section 4.2, R2, P92, SHA-256), "
            "see [a](docs/X_2.md) and `x = 3` and $\\alpha = 0.049$ and 2026-11-03; the next 15 minutes, "
            "the last-30-minute premium and point 1. The hybrid is 5.8% better (p 0.0006), slope -0.057, "
            "21,239 returns, p-values as small as 10^-256.")
    found = [tok for _, tok in tn.scan_numbers(text)]
    assert found == ["5.8", "0.0006", "-0.057", "21,239", "10^-256"]


def test_scan_numbers_list_markers_and_headings():
    text = "## 4. Results 12\n\n1. **First** has 942 days\n2. Second\n"
    assert [tok for _, tok in tn.scan_numbers(text)] == ["942"]


# ------------------------------------------------------------------------------------------
# Whole runs: status codes and outputs
# ------------------------------------------------------------------------------------------

def _registry(path: Path, rows: list[list[str]]) -> Path:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(tn.REGISTRY_COLUMNS)
        w.writerows(rows)
    return path


@pytest.fixture
def mini_repo(tmp_path):
    root = tmp_path / "repo"
    (root / "results" / "x").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "PITCH_PANDEY.md").write_text(
        "# Pitch\n\nThe hybrid is 5.8% better on 792 days, and 99.6% of rules lose. "
        "An unchecked 123.4 here. A 50-point strike.\n", encoding="utf-8")
    (root / "docs" / "D.md").write_text("Slope -0.069 (p 0.0006).\n", encoding="utf-8")
    write_csv(root / "results" / "x" / "s.csv", [
        {"group": "all", "pct": "-5.7988", "days": "792", "share": "0.0"},
        {"group": "expiry", "pct": "-6.6426", "days": "169", "share": "0.5"},
    ])
    write_csv(root / "results" / "x" / "r.csv", [{"coef": "-0.069512", "p": "0.000664"}])
    reg = _registry(root / "docs" / "number_registry.csv", [
        ["A1", "PITCH_PANDEY.md", "The hybrid is 5.8% better", "5.8", "neg", "results/x/s.csv", "csv:group=all@pct", ""],
        ["A2", "PITCH_PANDEY.md", "on 792 days", "792", "", "results/x/s.csv", "csv:group=all@days", ""],
        ["A3", "PITCH_PANDEY.md", "and 99.6% of rules lose", "99.6", "one_minus;x100", "results/x/s.csv",
         "csv:group=all@share", "file gives 100"],
        ["B1", "docs/D.md", "Slope -0.069 (p 0.0006)", "-0.069", "", "results/x/r.csv", "csv:@coef", ""],
        ["B2", "docs/D.md", "Slope -0.069 (p 0.0006)", "0.0006", "", "results/x/r.csv", "csv:@p", ""],
        ["C1", "docs/D.md", "a sentence that was deleted", "1.0", "", "results/x/r.csv", "csv:@p", ""],
        ["C2", "docs/D.md", "Slope -0.069 (p 0.0006)", "0.07", "", "results/x/r.csv", "csv:@p", "value not in snippet"],
        ["D1", "docs/D.md", "Slope -0.069 (p 0.0006)", "-0.069", "", "results/x/missing.csv", "csv:@coef", ""],
        ["D2", "docs/D.md", "Slope -0.069 (p 0.0006)", "-0.069", "", "results/x/r.csv", "csv:@nocol", ""],
        ["D3", "docs/D.md", "Slope -0.069 (p 0.0006)", "-0.069", "", "results/x/s.csv", "csv:@pct", "two rows"],
        ["E1", "docs/D.md", "Slope -0.069 (p 0.0006)", "-0.069", "", "", "UNTRACED", "no source"],
        ["P1", "PAPER", "falls from 359 to 72", "359", "", "results/x/s.csv", "csv:group=all@days", "wrong on purpose"],
    ])
    paper = tmp_path / "private" / "PAPER.md"
    paper.parent.mkdir()
    paper.write_text("Intro. It falls from 359 to 72 minutes, and 17.5 more.\n\n## References\n\nVol. 99, pp. 1 to 20.\n",
                     encoding="utf-8")
    return root, reg, paper


def _statuses(rows):
    return {r["claim_id"]: r["status"] for r in rows}


def test_status_codes(mini_repo, tmp_path):
    root, reg, paper = mini_repo
    rows = tn.run(reg, root, root / "results" / "number_trace", paper, quiet=True)
    st = _statuses(rows)
    assert st == {
        "A1": "MATCH", "A2": "MATCH", "A3": "MISMATCH",
        "B1": "MISMATCH",  # -0.0695 rounds to -0.070
        "B2": "MISMATCH",  # 0.000664 rounds to 0.0007
        "C1": "TEXT_MISSING", "C2": "TEXT_MISSING",
        "D1": "SOURCE_MISSING", "D2": "SOURCE_MISSING", "D3": "SOURCE_MISSING",
        "E1": "UNTRACED",
        "P1": "MISMATCH",  # the paper snippet is found, the file value 792 is not 359
    }
    out = root / "results" / "number_trace"
    with open(out / "trace.csv", encoding="utf-8") as fh:
        written = list(csv.DictReader(fh))
    assert [r["claim_id"] for r in written] == [r["claim_id"] for r in rows]
    assert set(written[0]) == set(tn.TRACE_COLUMNS)
    assert all(r["status"] in tn.STATUSES for r in written)
    assert {r["claim_id"]: r["text_found"] for r in written}["C2"] == "no"


def test_without_paper_the_paper_text_is_not_checked(mini_repo):
    root, reg, _ = mini_repo
    rows = {r["claim_id"]: r for r in tn.run(reg, root, root / "out", None, quiet=True)}
    assert rows["P1"]["text_found"] == "not checked"
    assert "not checked" in rows["P1"]["detail"]


def test_uncovered_pitch_csv(mini_repo):
    root, reg, paper = mini_repo
    out = root / "results" / "number_trace"
    tn.run(reg, root, out, paper, quiet=True)
    with open(out / "uncovered_pitch.csv", encoding="utf-8") as fh:
        unc = list(csv.DictReader(fh))
    assert [u["number"] for u in unc] == ["123.4", "50"]
    assert unc[0]["line"] == "3"


def test_paper_uncovered_goes_to_stdout_only(mini_repo, capsys):
    root, reg, paper = mini_repo
    out = root / "results" / "number_trace"
    tn.run(reg, root, out, paper)
    printed = capsys.readouterr().out
    assert "17.5" in printed           # uncovered paper number before the references
    assert "pp. 1 to 20" not in printed  # nothing after "## References"
    assert sorted(p.name for p in out.iterdir()) == ["trace.csv", "uncovered_pitch.csv"]


def test_paper_uncovered_file_must_be_outside_repo(mini_repo, tmp_path):
    root, reg, paper = mini_repo
    with pytest.raises(SystemExit):
        tn.run(reg, root, root / "out", paper, paper_uncovered_out=root / "paper_uncovered.csv", quiet=True)
    assert not (root / "paper_uncovered.csv").exists()
    assert not (root / "out").exists()  # refused before anything is written
    outside = tmp_path / "scratch" / "paper_uncovered.csv"
    tn.run(reg, root, root / "out", paper, paper_uncovered_out=outside, quiet=True)
    with open(outside, encoding="utf-8") as fh:
        assert [u["number"] for u in csv.DictReader(fh)] == ["72", "17.5"]


def test_registry_validation(tmp_path):
    reg = _registry(tmp_path / "r.csv", [["A", "x.md", "q", "1", "", "f.csv", "csv:@#count", ""]] * 2)
    with pytest.raises(ValueError, match="duplicate"):
        tn.load_registry(reg)
    bad = tmp_path / "bad.csv"
    bad.write_text("claim_id,document\nA,x.md\n", encoding="utf-8")
    with pytest.raises(ValueError, match="lacks columns"):
        tn.load_registry(bad)


def test_cli(mini_repo, tmp_path, capsys):
    root, reg, paper = mini_repo
    out = tmp_path / "cli_out"
    assert tn.main(["--registry", str(reg), "--root", str(root), "--out", str(out), "--quiet"]) == 0
    assert (out / "trace.csv").exists()
    with pytest.raises(SystemExit):
        tn.main(["--registry", str(reg), "--root", str(root), "--out", str(out), "--paper", str(tmp_path / "nope.md")])
