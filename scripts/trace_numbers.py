#!/usr/bin/env python
"""Number tracing check (P98): do the numbers printed in the write-ups match the results files?

Reads a hand-written registry (``docs/number_registry.csv``). Each row names one printed number:
the document it appears in, a short quoted snippet around it, the value as printed, how to compare
it, the results file it should come from and a selector that says how to read the value from that
file. For every row the script checks

(a) that the quoted snippet still appears in the document (whitespace and line breaks are ignored,
    and the printed value must sit inside the snippet), and
(b) that the value read from the results file, after any transform the registry names, rounds to
    the printed value.

It writes one row per registry entry to ``results/number_trace/trace.csv`` with a status:

* ``MATCH``: the snippet is in the document and the file value rounds to the printed value.
* ``MISMATCH``: the snippet is in the document but the file value does not round to it.
* ``TEXT_MISSING``: the snippet (or the printed value inside it) is no longer in the document.
* ``SOURCE_MISSING``: the results file is missing, or the selector finds no value in it
  (no row, several rows where one was expected, a missing column or key, no regex match).
* ``UNTRACED``: the registry records that no results file for this number was found.

It also scans ``PITCH_PANDEY.md`` (and, if given, the working paper's main text) for numbers that no
registry entry covers. The pitch list goes to ``results/number_trace/uncovered_pitch.csv``. The
paper is private: its uncovered list is printed to stdout, or written to ``--paper-uncovered-out``,
which must lie outside the repository. Registry rows whose document is ``PAPER`` refer to the file
passed with ``--paper``; without it their values are still checked but their text is reported as
"not checked".

Selectors (column ``selector``; several selectors can be joined with `` || ``, see the transforms):

* ``csv:<filter>@<column>``: a CSV (or ``.csv.gz``) file. ``<filter>`` is ``key=value`` pairs joined
  by ``&`` (an empty filter selects every row). Values are compared as text first, then as numbers,
  so ``window=6 months`` and ``n=929`` both work; ``key<value``, ``<=``, ``>``, ``>=`` compare
  numbers and ``key!=value`` excludes. ``<column>`` is a column name; the filter must
  then select exactly one row. ``#count`` returns the number of selected rows, and ``#sum:col``,
  ``#mean:col``, ``#min:col``, ``#max:col`` aggregate a column over them (``True`` counts as 1);
  ``#topshare:col:k`` is the share of the column's total that its k largest values make up.
* ``json:<key.path>``: a JSON file; keys are separated by ``.`` (write ``\\.`` for a dot inside a
  key) and ``[n]`` indexes a list.
* ``regex:<pattern>``: a text file (for example a ``*_stdout.txt`` log); the first capture group of
  the first match (or the whole match if there is no group).
* ``derived:<name>?k=v&k=v``: a small named computation on a ``forecasts_<target>.csv`` file, for
  numbers the code computes only in passing: ``daily_win_share`` (share of days on which a model's
  daily mean QLIKE is below a benchmark's), ``top_k_sq_error_share`` (share of a model's total squared
  error from its k largest errors) and ``minute_gain_share`` (share of a model's total QLIKE gain over
  a benchmark that is made at one forecast minute).

``results_file`` holds one path (relative to the repository root) or, for selectors joined with
`` || ``, one path per selector joined the same way.

Comparison rules (column ``rounding_or_tolerance``; tokens separated by ``;``):

* Mode (one of): ``round`` (default: the file value must round to the printed value at the printed
  precision, ties count as a match, so 5.84 matches 5.8 and -0.0574 matches -0.057 but 5.86 does not
  match 5.8), ``lt`` / ``le`` / ``gt`` (printed bounds such as "p < 0.001"), ``abs=X`` and ``rel=X``
  (explicit tolerance), ``exact``, ``oom`` (same power of ten, for "as small as 10^-256"), ``str``
  (text equality) and ``date`` (the same calendar date, so "14 November 2022" matches "2022-11-14").
  A printed value may also be a quantity in words ("all", "none", "six", "one in five"); it is read
  as an exact number, so a word that the text qualifies with "about" needs ``abs=X`` or ``rel=X``.
* Transforms of the file value, applied left to right: ``x100`` (fraction to percent; used only where
  the registry says so), ``neg``, ``abs``, ``one_minus``, ``ratio_pct`` ((x - 1) x 100),
  ``expm1_pct`` ((e^x - 1) x 100), ``half``, ``mul=X`` and ``divby=X`` (for example ``divby=60`` for
  seconds to minutes). To combine several selected values (applied first): ``sum``, ``diff`` (first
  minus second), ``div`` (first over second), ``mul`` (product), ``max``, ``min``.

Usage::

    python scripts/trace_numbers.py                                    # repository documents only
    python scripts/trace_numbers.py --paper /path/to/WORKING_PAPER.md  # also the private paper

The script never writes anything derived from the paper into the repository except the registry's
own short snippets (which ``trace.csv`` repeats).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PAPER_TOKEN = "PAPER"
STATUSES = ("MATCH", "MISMATCH", "TEXT_MISSING", "SOURCE_MISSING", "UNTRACED")
REGISTRY_COLUMNS = ["claim_id", "document", "quoted_text", "value", "rounding_or_tolerance",
                    "results_file", "selector", "note"]
TRACE_COLUMNS = ["claim_id", "document", "quoted_text", "printed_value", "file_value", "compared_value",
                 "rule", "results_file", "selector", "text_found", "status", "detail", "note"]
MODES = ("round", "lt", "le", "gt", "exact", "oom", "str", "date")
TRANSFORMS = ("x100", "neg", "abs", "one_minus", "ratio_pct", "expm1_pct", "half")
PARAM_TRANSFORMS = ("mul", "divby")
COMBINERS = ("sum", "diff", "div", "mul", "max", "min")
_FILTER_OPS = ("<=", ">=", "!=", "<", ">", "=")


class SourceError(Exception):
    """The results file or the value inside it could not be found."""


# --------------------------------------------------------------------------------------------
# Printed numbers
# --------------------------------------------------------------------------------------------

_POW10 = re.compile(r"^([+-]?)10\^([+-]?\d+)$")


def normalise_minus(s: str) -> str:
    # U+2212 (minus sign), U+2013 and U+2014 (typographic dashes) all read as a minus
    return s.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")


_NUMBER_WORDS = {"none": 0, "zero": 0, "all": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                 "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
_WORD_RATIO = re.compile(r"^(\w+) (?:in|out of) (\w+)$")
_WORD_PLACE = -12  # a quantity in words is exact


def _word_number(word: str) -> Decimal | None:
    if word in _NUMBER_WORDS:
        return Decimal(_NUMBER_WORDS[word])
    return Decimal(word) if word.isdigit() else None


def parse_words(text: str) -> Decimal | None:
    """A quantity written in words: "all" (1), "none" (0), "six" (6), "one in five" (0.2)."""
    t = re.sub(r"\s+", " ", str(text).strip().lower())
    m = _WORD_RATIO.match(t)
    if m:
        a, b = _word_number(m.group(1)), _word_number(m.group(2))
        return a / b if a is not None and b else None
    if t.isdigit():
        return None
    return _word_number(t)


def parse_printed(text: str) -> tuple[Decimal, int]:
    """Value and precision of a printed number.

    Returns the value and the decimal exponent of its last printed digit, so that "5.8" gives
    (5.8, -1), "21,239" gives (21239, 0), "1.4e-12" gives (1.4e-12, -13) and "10^-256" gives
    (1e-256, -256). A trailing "%" is ignored (percent against fraction is a registry transform).
    A quantity in words ("all", "none", "six", "one in five") is read as an exact number (1, 0, 6,
    0.2), so "about one in five" needs an explicit ``abs=`` or ``rel=`` tolerance.
    """
    w = parse_words(text)
    if w is not None:
        return w, _WORD_PLACE
    s = normalise_minus(str(text)).strip().replace(" ", "")
    s = s.rstrip("%")
    if s.startswith("<") or s.startswith(">"):
        s = s[1:]
    m = _POW10.match(s)
    if m:
        exp = int(m.group(2))
        sign = -1 if m.group(1) == "-" else 1
        return Decimal(sign) * Decimal(10) ** exp, exp
    s = s.replace(",", "")
    try:
        d = Decimal(s)
    except InvalidOperation as exc:
        raise ValueError(f"not a printed number: {text!r}") from exc
    return d, d.as_tuple().exponent


_DATE_FORMATS = ("%Y-%m-%d", "%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y", "%d %B, %Y")


def parse_date(text: str):
    """A date printed in words ("14 November 2022") or ISO ("2022-11-14", optionally with a time)."""
    s = re.sub(r"\s+", " ", str(text).strip())
    s = re.sub(r"[T ]\d{2}:\d{2}(:\d{2})?.*$", "", s) if re.match(r"^\d{4}-\d{2}-\d{2}[T ]", s) else s
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"not a date: {text!r}")


def precision_half_unit(place: int) -> float:
    return 0.5 * 10.0 ** place


# --------------------------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------------------------

@dataclass
class Rule:
    mode: str = "round"
    tol: float | None = None
    transforms: list[str] = field(default_factory=list)
    combiner: str | None = None


def parse_rule(spec: str) -> Rule:
    rule = Rule()
    for tok in [t.strip() for t in (spec or "").split(";") if t.strip()]:
        if tok.startswith("abs="):
            rule.mode, rule.tol = "abs", float(tok[4:])
        elif tok.startswith("rel="):
            rule.mode, rule.tol = "rel", float(tok[4:])
        elif tok in MODES:
            rule.mode = tok
        elif tok in TRANSFORMS:
            rule.transforms.append(tok)
        elif tok in COMBINERS:
            rule.combiner = tok
        elif tok.partition("=")[0] in PARAM_TRANSFORMS and tok.partition("=")[2]:
            float(tok.partition("=")[2])  # fail early on a bad number
            rule.transforms.append(tok)
        else:
            raise ValueError(f"unknown rule token {tok!r}")
    return rule


def combine(values: list[float], combiner: str | None) -> float:
    if len(values) == 1 and combiner is None:
        return values[0]
    if combiner is None:
        raise ValueError("several selectors need a combiner (sum, diff, div, mul, max or min)")
    if combiner == "sum":
        return float(sum(values))
    if combiner == "mul":
        return float(math.prod(values))
    if combiner == "max":
        return float(max(values))
    if combiner == "min":
        return float(min(values))
    if len(values) != 2:
        raise ValueError(f"{combiner} needs exactly two selectors")
    a, b = values
    return a - b if combiner == "diff" else a / b


def apply_transforms(x: float, transforms: list[str]) -> float:
    for t in transforms:
        if t == "x100":
            x = x * 100.0
        elif t == "neg":
            x = -x
        elif t == "abs":
            x = abs(x)
        elif t == "one_minus":
            x = 1.0 - x
        elif t == "ratio_pct":
            x = (x - 1.0) * 100.0
        elif t == "expm1_pct":
            x = math.expm1(x) * 100.0
        elif t == "half":
            x = x / 2.0
        elif t.startswith("mul="):
            x = x * float(t[4:])
        elif t.startswith("divby="):
            x = x / float(t[6:])
    return x


def compare(printed: str, file_value, rule: Rule) -> tuple[bool, str]:
    """True if the (transformed) file value agrees with the printed value under the rule."""
    if rule.mode == "str":
        ok = str(file_value).strip() == str(printed).strip()
        return ok, "text equal" if ok else "text differs"
    if rule.mode == "date":
        a, b = parse_date(str(printed)), parse_date(str(file_value))
        ok = a == b
        return ok, "same date" if ok else f"dates differ ({a} and {b})"
    p, place = parse_printed(printed)
    pf = float(p)
    f = float(file_value)
    if math.isnan(f):
        return False, "file value is NaN"
    if rule.mode == "round":
        half = precision_half_unit(place)
        ok = abs(f - pf) <= half * (1 + 1e-9) + 1e-15 * max(1.0, abs(pf))
        return ok, f"|file - printed| = {abs(f - pf):.3g}, allowed {half:.3g}"
    if rule.mode == "lt":
        return f < pf, f"file {'<' if f < pf else '>='} printed bound"
    if rule.mode == "le":
        return f <= pf, f"file {'<=' if f <= pf else '>'} printed bound"
    if rule.mode == "gt":
        return f > pf, f"file {'>' if f > pf else '<='} printed bound"
    if rule.mode == "exact":
        return f == pf, "equal" if f == pf else "not equal"
    if rule.mode == "abs":
        return abs(f - pf) <= rule.tol, f"|file - printed| = {abs(f - pf):.3g}, allowed {rule.tol}"
    if rule.mode == "rel":
        allowed = rule.tol * abs(pf)
        return abs(f - pf) <= allowed, f"|file - printed| = {abs(f - pf):.3g}, allowed {allowed:.3g}"
    if rule.mode == "oom":
        if f == 0 or pf == 0:
            return False, "zero has no order of magnitude"
        a, b = math.floor(math.log10(abs(f))), math.floor(math.log10(abs(pf)))
        return a == b, f"powers of ten {a} and {b}"
    raise ValueError(rule.mode)


# --------------------------------------------------------------------------------------------
# Selectors
# --------------------------------------------------------------------------------------------

def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return open(path, encoding="utf-8", newline="")


_CSV_CACHE: dict[Path, tuple[list[str], list[dict[str, str]]]] = {}


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if path not in _CSV_CACHE:
        csv.field_size_limit(sys.maxsize)
        with _open_text(path) as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
            _CSV_CACHE[path] = (list(reader.fieldnames or []), rows)
    return _CSV_CACHE[path]


def _cell_matches(cell: str, want: str) -> bool:
    if cell is None:
        return False
    if cell.strip() == want.strip():
        return True
    try:
        return float(cell) == float(want)
    except ValueError:
        return False


def _as_number(cell: str) -> float:
    c = (cell or "").strip()
    if c in ("True", "true"):
        return 1.0
    if c in ("False", "false"):
        return 0.0
    if c == "":
        raise SourceError("empty cell")
    try:
        return float(c)
    except ValueError as exc:
        raise SourceError(f"not a number: {c!r}") from exc


def parse_csv_selector(body: str) -> tuple[list[tuple[str, str, str]], str]:
    """Filters as (column, operator, value) triples, and the column or aggregate after '@'."""
    if "@" not in body:
        raise ValueError(f"csv selector needs '@<column>': {body!r}")
    filt, col = body.rsplit("@", 1)
    triples = []
    for part in [p for p in filt.split("&") if p.strip()]:
        m = re.match(r"^([^<>=!]+)(<=|>=|!=|<|>|=)(.*)$", part)
        if not m:
            raise ValueError(f"filter needs key=value (or <, <=, >, >=, !=): {part!r}")
        triples.append((m.group(1).strip(), m.group(2), m.group(3).strip()))
    return triples, col.strip()


def _filter_ok(cell: str, op: str, want: str) -> bool:
    if op == "=":
        return _cell_matches(cell, want)
    if op == "!=":
        return not _cell_matches(cell, want)
    try:
        a, b = float(cell), float(want)
    except (TypeError, ValueError):
        return False
    if math.isnan(a):
        return False
    return {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b}[op]


def select_csv(path: Path, body: str):
    triples, col = parse_csv_selector(body)
    fields, rows = read_csv_rows(path)
    for k, _, _ in triples:
        if k not in fields:
            raise SourceError(f"no column {k!r} in {path.name}")
    hit = [r for r in rows if all(_filter_ok(r.get(k), op, v) for k, op, v in triples)]
    if col == "#count":
        return float(len(hit))
    if col.startswith("#"):
        agg, _, c = col[1:].partition(":")
        k = None
        if agg == "topshare":
            c, _, k = c.rpartition(":")
            k = int(k)
        if c not in fields:
            raise SourceError(f"no column {c!r} in {path.name}")
        if not hit:
            raise SourceError("filter selects no row")
        vals = [_as_number(r[c]) for r in hit if (r.get(c) or "").strip() != ""]
        if not vals:
            raise SourceError(f"column {c!r} is empty in every selected row")
        if agg == "sum":
            return float(sum(vals))
        if agg == "mean":
            return float(sum(vals) / len(vals))
        if agg == "min":
            return float(min(vals))
        if agg == "max":
            return float(max(vals))
        if agg == "topshare":
            return float(sum(sorted(vals, reverse=True)[:k]) / sum(vals))
        raise ValueError(f"unknown aggregate {agg!r}")
    if col not in fields:
        raise SourceError(f"no column {col!r} in {path.name}")
    if len(hit) != 1:
        raise SourceError(f"filter selects {len(hit)} rows, expected 1")
    return hit[0][col]


def select_json(path: Path, body: str):
    with open(path, encoding="utf-8") as fh:
        obj = json.load(fh)
    keys = re.split(r"(?<!\\)\.", body)
    for raw in keys:
        key = raw.replace("\\.", ".")
        m = re.fullmatch(r"(.*?)((?:\[\d+\])*)", key)
        name, idx = m.group(1), m.group(2)
        if name:
            if not isinstance(obj, dict) or name not in obj:
                raise SourceError(f"no key {name!r} in {path.name}")
            obj = obj[name]
        for i in re.findall(r"\[(\d+)\]", idx):
            if not isinstance(obj, list) or int(i) >= len(obj):
                raise SourceError(f"no index [{i}] in {path.name}")
            obj = obj[int(i)]
    if isinstance(obj, (dict, list)):
        raise SourceError("key path ends at a container, not a value")
    return obj


def select_regex(path: Path, body: str):
    text = path.read_text(encoding="utf-8")
    m = re.search(body, text, flags=re.MULTILINE)
    if not m:
        raise SourceError(f"pattern not found in {path.name}")
    return m.group(1) if m.groups() else m.group(0)


def _daily_win_share(path: Path, params: dict[str, str]) -> float:
    """Share of days on which ``model``'s daily mean QLIKE is below ``bench``'s.

    QLIKE as in ``niftygex.evaluation.qlike`` (both arguments floored at 1e-16).
    """
    model, bench = params.get("model", "hybrid_base"), params.get("bench", "mcsgarch")
    _, rows = read_csv_rows(path)
    if not rows or model not in rows[0] or bench not in rows[0]:
        raise SourceError(f"columns {model!r} or {bench!r} missing in {path.name}")

    q = _qlike
    per_day: dict[str, list[float]] = {}
    for r in rows:
        y = float(r["y"])
        acc = per_day.setdefault(r["date"], [0.0, 0.0, 0])
        acc[0] += q(y, float(r[model]))
        acc[1] += q(y, float(r[bench]))
        acc[2] += 1
    wins = sum(1 for a in per_day.values() if a[0] / a[2] < a[1] / a[2])
    return wins / len(per_day)


def _qlike(y: float, f: float) -> float:
    y, f = max(y, 1e-16), max(f, 1e-16)
    return y / f - math.log(y / f) - 1


def _top_k_sq_error_share(path: Path, params: dict[str, str]) -> float:
    """Share of ``model``'s total squared forecast error that comes from its ``k`` largest errors."""
    model, k = params.get("model", "mcsgarch"), int(params.get("k", "10"))
    _, rows = read_csv_rows(path)
    if not rows or model not in rows[0]:
        raise SourceError(f"column {model!r} missing in {path.name}")
    se = sorted(((float(r[model]) - float(r["y"])) ** 2 for r in rows), reverse=True)
    return sum(se[:k]) / sum(se)


def _minute_gain_share(path: Path, params: dict[str, str]) -> float:
    """Share of ``model``'s total QLIKE gain over ``bench`` (summed over all forecasts) made at ``minute``.

    ``minute`` counts from the open, so 0 is the 09:15 forecast.
    """
    model, bench = params.get("model", "hybrid_base"), params.get("bench", "mcsgarch")
    minute = float(params.get("minute", "0"))
    _, rows = read_csv_rows(path)
    if not rows or model not in rows[0] or bench not in rows[0]:
        raise SourceError(f"columns {model!r} or {bench!r} missing in {path.name}")
    total = at = 0.0
    for r in rows:
        y = float(r["y"])
        g = _qlike(y, float(r[bench])) - _qlike(y, float(r[model]))
        total += g
        if float(r["minute"]) == minute:
            at += g
    return at / total


DERIVED = {"daily_win_share": _daily_win_share, "top_k_sq_error_share": _top_k_sq_error_share,
           "minute_gain_share": _minute_gain_share}


def select_one(path: Path, selector: str):
    kind, _, body = selector.partition(":")
    kind = kind.strip()
    if not path.exists():
        raise SourceError(f"file not found: {path}")
    if kind == "csv":
        return select_csv(path, body)
    if kind == "json":
        return select_json(path, body)
    if kind == "regex":
        return select_regex(path, body)
    if kind == "derived":
        name, _, query = body.partition("?")
        params = dict(p.split("=", 1) for p in query.split("&") if p)
        if name not in DERIVED:
            raise ValueError(f"unknown derived computation {name!r}")
        return DERIVED[name](path, params)
    raise ValueError(f"unknown selector kind {kind!r}")


def split_multi(s: str) -> list[str]:
    return [p.strip() for p in (s or "").split("||")]


def resolve_value(results_file: str, selector: str, rule: Rule, root: Path):
    """Read the file value(s), combine and transform them. Returns (raw text, compared value)."""
    sels = split_multi(selector)
    files = split_multi(results_file)
    if len(files) == 1:
        files = files * len(sels)
    if len(files) != len(sels):
        raise ValueError("results_file and selector have different numbers of parts")
    raw = [select_one(root / f, s) for f, s in zip(files, sels)]
    if rule.mode in ("str", "date"):
        if len(raw) != 1:
            raise ValueError(f"{rule.mode} mode takes one selector")
        return str(raw[0]), str(raw[0]).strip()
    nums = []
    for r in raw:
        try:
            nums.append(float(r))
        except (TypeError, ValueError) as exc:
            raise SourceError(f"not a number: {r!r}") from exc
    value = apply_transforms(combine(nums, rule.combiner), rule.transforms)
    raw_text = " | ".join(repr(float(n)) for n in nums)
    return raw_text, value


# --------------------------------------------------------------------------------------------
# Documents
# --------------------------------------------------------------------------------------------

def normalise_ws(s: str) -> str:
    return re.sub(r"\s+", " ", normalise_minus(s)).strip()


def _norm_with_map(text: str) -> tuple[str, list[int]]:
    """Whitespace-collapsed text plus, for each output character, its index in the original."""
    out, idx, prev_space = [], [], False
    for i, ch in enumerate(normalise_minus(text)):
        if ch.isspace():
            if not prev_space:
                out.append(" ")
                idx.append(i)
            prev_space = True
        else:
            out.append(ch)
            idx.append(i)
            prev_space = False
    return "".join(out), idx


@dataclass
class Document:
    path: Path
    text: str

    def __post_init__(self):
        self.norm, self.map = _norm_with_map(self.text)

    def find_all(self, snippet: str) -> list[int]:
        """Start positions (in the normalised text) of every occurrence of the snippet."""
        s = normalise_ws(snippet)
        out, start = [], 0
        while s:
            k = self.norm.find(s, start)
            if k < 0:
                break
            out.append(k)
            start = k + 1
        return out


def value_offset_in_snippet(snippet: str, value: str) -> int | None:
    """Where the printed value sits inside the snippet, as a number token (not inside a longer one)."""
    s, v = normalise_ws(snippet), normalise_minus(str(value).strip())
    candidates = {v, v.lstrip("+")}
    if v.startswith("-"):
        candidates.add(v[1:])
    for c in sorted(candidates, key=len, reverse=True):
        for m in re.finditer(re.escape(c), s):
            before = s[m.start() - 1] if m.start() > 0 else " "
            after = s[m.end()] if m.end() < len(s) else " "
            if before.isdigit() or before == "." or after.isdigit():
                continue
            if (c[:1].isalpha() and before.isalpha()) or (c[-1:].isalpha() and after.isalpha()):
                continue  # a word value ("all") must be a whole word, not part of "small"
            if after == "." and m.end() + 1 < len(s) and s[m.end() + 1].isdigit():
                continue
            return m.start()
    return None


# --------------------------------------------------------------------------------------------
# Registry and checking
# --------------------------------------------------------------------------------------------

def load_registry(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in REGISTRY_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"registry lacks columns {missing}")
        rows = [{k: (v or "") for k, v in r.items()} for r in reader]
    ids = [r["claim_id"] for r in rows]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise ValueError(f"duplicate claim_id: {dup}")
    return rows


def is_untraced(entry: dict[str, str]) -> bool:
    sel = entry["selector"].strip()
    return entry["results_file"].strip() == "" or sel == "" or sel.upper() == "UNTRACED"


def check_entry(entry: dict[str, str], docs: dict[str, Document | None], root: Path) -> dict[str, str]:
    out = {
        "claim_id": entry["claim_id"], "document": entry["document"], "quoted_text": entry["quoted_text"],
        "printed_value": entry["value"], "file_value": "", "compared_value": "",
        "rule": entry["rounding_or_tolerance"] or "round", "results_file": entry["results_file"],
        "selector": entry["selector"], "text_found": "", "status": "", "detail": "", "note": entry["note"],
    }
    doc = docs.get(entry["document"])
    text_problem = None
    if doc is None:
        out["text_found"] = "not checked"
    else:
        hits = doc.find_all(entry["quoted_text"])
        if not hits:
            out["text_found"] = "no"
            text_problem = "quoted text not found in the document"
        elif value_offset_in_snippet(entry["quoted_text"], entry["value"]) is None:
            out["text_found"] = "no"
            text_problem = "printed value is not inside the quoted text"
        else:
            out["text_found"] = "yes"

    if is_untraced(entry):
        out["status"] = "UNTRACED"
        out["detail"] = text_problem or ""
        return out

    try:
        rule = parse_rule(entry["rounding_or_tolerance"])
        raw, value = resolve_value(entry["results_file"], entry["selector"], rule, root)
        out["file_value"] = raw
        out["compared_value"] = value if isinstance(value, str) else f"{value:.10g}"
        ok, why = compare(entry["value"], value, rule)
    except SourceError as exc:
        out["status"] = "TEXT_MISSING" if text_problem else "SOURCE_MISSING"
        out["detail"] = "; ".join(x for x in (text_problem, f"source: {exc}") if x)
        return out
    except (ValueError, KeyError) as exc:
        out["status"] = "TEXT_MISSING" if text_problem else "SOURCE_MISSING"
        out["detail"] = "; ".join(x for x in (text_problem, f"registry problem: {exc}") if x)
        return out

    if text_problem:
        out["status"] = "TEXT_MISSING"
        out["detail"] = f"{text_problem}; value check: {'agrees' if ok else 'differs'} ({why})"
    else:
        out["status"] = "MATCH" if ok else "MISMATCH"
        out["detail"] = why if doc is not None else f"value only, document text not checked; {why}"
    return out


# --------------------------------------------------------------------------------------------
# Coverage scan
# --------------------------------------------------------------------------------------------

_MONTHS = ("January|February|March|April|May|June|July|August|September|October|November|December|"
           "Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec")
_NUM = re.compile(r"(?<![A-Za-z0-9_.^])(?<![A-Za-z]-)[+-]?(?:10\^[+-]?\d+|(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
                  r"(?:e[+-]?\d+)?)(?![A-Za-z0-9_^])")


def _mask(text: str, pattern: str, flags=0) -> str:
    return re.sub(pattern, lambda m: " " * len(m.group(0)), text, flags=flags)


def scan_numbers(text: str) -> list[tuple[int, str]]:
    """Positions and text of the numbers worth tracing in a markdown text.

    Leaves out dates, years, times, list numbers, section/table/figure references, link targets,
    code, math, HTML comments, identifiers such as R2, P92 or SHA-256 (a digit attached to a letter)
    and the forecast horizons "15 minutes", "30-minute" and "60 minutes", which are design settings.
    """
    t = normalise_minus(text)
    t = _mask(t, r"<!--.*?-->", re.S)
    t = _mask(t, r"\$\$.*?\$\$", re.S)
    t = _mask(t, r"\$[^$\n]*\$")
    t = _mask(t, r"`[^`\n]*`")
    t = _mask(t, r"\]\([^)]*\)")
    t = _mask(t, r"\b\d{4}-\d{2}-\d{2}\b")
    t = _mask(t, r"\[[^\]\n]*\.(?:md|csv|py|png|json|txt)\]")
    t = _mask(t, r"!\[[^\]]*\]")
    t = _mask(t, r"\[\^[^\]]*\]")
    t = _mask(t, rf"\b\d{{1,2}}(?:\s+and\s+\d{{1,2}})?\s+(?:{_MONTHS})\b(?:\s+(?:19|20)\d{{2}})?")
    t = _mask(t, rf"\b(?:{_MONTHS})\s+\d{{1,2}}(?:,\s*(?:19|20)\d{{2}})?\b")
    t = _mask(t, rf"\b(?:{_MONTHS})\s+(?:19|20)\d{{2}}\b")
    t = _mask(t, r"\b(?:19|20)\d{2}(?:H[12])?\b(?![.,]\d)")
    t = _mask(t, r"\b\d{1,2}:\d{2}\b")
    t = _mask(t, r"\b(?:15|30|60)(?:\s+and\s+(?:15|30|60))?[- ]minutes?\b")
    t = _mask(t, r"(?m)^\s*(?:[-*]\s+)?\d+\.(?=\s)")
    t = _mask(t, r"(?m)^#+.*$")
    t = _mask(t, r"\b(?:Table|Tables|Section|Sections|Figure|Figures|Appendix|Iteration|Iterations|Panel|"
                 r"Result|Results|Contributions?|Finding|Findings|item|items|step|Step|Event|event|Week|week|point)"
                 r"\s+[A-Z]?\d+(?:\.\d+)*(?:\s*(?:,|and|to)\s*[A-Z]?\d+(?:\.\d+)*)*")
    return [(m.start(), m.group(0)) for m in _NUM.finditer(t)]


def covered_positions(doc: Document, entries: list[dict[str, str]]) -> set[int]:
    """Original-text start positions of the numbers that registry entries cover."""
    cov: set[int] = set()
    for e in entries:
        off = value_offset_in_snippet(e["quoted_text"], e["value"])
        if off is None:
            continue
        for k in doc.find_all(e["quoted_text"]):
            j = k + off
            if j < len(doc.map):
                cov.add(doc.map[j])
    return cov


def uncovered_numbers(doc: Document, entries: list[dict[str, str]], stop_at: str | None = None) -> list[dict]:
    text = doc.text
    if stop_at:
        k = text.find(stop_at)
        if k >= 0:
            text = text[:k]
    cov = covered_positions(doc, entries)
    out = []
    for pos, tok in scan_numbers(text):
        starts = {pos, pos + 1} if tok[:1] in "+-" else {pos}
        if starts & cov:
            continue
        line = text.count("\n", 0, pos) + 1
        ls = text.rfind("\n", 0, pos) + 1
        le = text.find("\n", pos)
        le = len(text) if le < 0 else le
        a, b = max(ls, pos - 60), min(le, pos + len(tok) + 60)
        out.append({"line": line, "number": tok, "context": normalise_ws(text[a:b])})
    return out


# --------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------

def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def run(registry: Path, root: Path, out_dir: Path, paper: Path | None,
        paper_uncovered_out: Path | None = None, quiet: bool = False) -> list[dict[str, str]]:
    if paper_uncovered_out is not None and _inside(paper_uncovered_out, root):
        raise SystemExit("--paper-uncovered-out must be outside the repository (the paper is private)")
    entries = load_registry(registry)
    docs: dict[str, Document | None] = {}
    for name in sorted({e["document"] for e in entries} | {"PITCH_PANDEY.md"}):
        if name == PAPER_TOKEN:
            docs[name] = Document(paper, paper.read_text(encoding="utf-8")) if paper else None
            continue
        p = root / name
        docs[name] = Document(p, p.read_text(encoding="utf-8")) if p.exists() else Document(p, "")

    rows = [check_entry(e, docs, root) for e in entries]
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "trace.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=TRACE_COLUMNS)
        w.writeheader()
        w.writerows(rows)

    pitch_entries = [e for e in entries if e["document"] == "PITCH_PANDEY.md"]
    unc = uncovered_numbers(docs["PITCH_PANDEY.md"], pitch_entries)
    with open(out_dir / "uncovered_pitch.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["line", "number", "context"])
        w.writeheader()
        w.writerows(unc)

    counts = {s: sum(r["status"] == s for r in rows) for s in STATUSES}
    if not quiet:
        print(f"{len(rows)} registry entries: " + ", ".join(f"{s} {n}" for s, n in counts.items()))
        for r in rows:
            if r["status"] not in ("MATCH", "UNTRACED"):
                print(f"  {r['status']:<14} {r['claim_id']:<10} printed {r['printed_value']!s:<10} "
                      f"file {r['compared_value']!s:<14} {r['detail']}")
        print(f"pitch: {len(unc)} numbers without a registry entry -> {out_dir / 'uncovered_pitch.csv'}")

    if paper is not None and docs.get(PAPER_TOKEN) is not None:
        paper_entries = [e for e in entries if e["document"] == PAPER_TOKEN]
        punc = uncovered_numbers(docs[PAPER_TOKEN], paper_entries, stop_at="\n## References")
        if paper_uncovered_out is not None:
            paper_uncovered_out.parent.mkdir(parents=True, exist_ok=True)
            with open(paper_uncovered_out, "w", encoding="utf-8", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=["line", "number", "context"])
                w.writeheader()
                w.writerows(punc)
            if not quiet:
                print(f"paper main text: {len(punc)} numbers without a registry entry -> {paper_uncovered_out}")
        elif not quiet:
            print(f"paper main text: {len(punc)} numbers without a registry entry (not saved):")
            for u in punc:
                print(f"  line {u['line']:>4}  {u['number']:<12} {u['context']}")
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--registry", type=Path, default=REPO / "docs" / "number_registry.csv")
    ap.add_argument("--root", type=Path, default=REPO, help="repository root (documents and results)")
    ap.add_argument("--out", type=Path, default=None, help="output folder (default results/number_trace)")
    ap.add_argument("--paper", type=Path, default=None, help="path to the private working paper (read only)")
    ap.add_argument("--paper-uncovered-out", type=Path, default=None,
                    help="optional CSV for the paper's uncovered numbers; must be outside the repository")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    out = a.out or (a.root / "results" / "number_trace")
    if a.paper is not None and not a.paper.exists():
        raise SystemExit(f"paper not found: {a.paper}")
    run(a.registry, a.root, out, a.paper, a.paper_uncovered_out, a.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
