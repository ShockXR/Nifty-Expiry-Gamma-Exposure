"""SYNTHETIC tests for the expiry-schedule detector of experiments/expiry_event_study.py after the
holiday-shift fix: a sample that opens with holiday-shifted expiries must not seed a false regime,
an isolated shifted expiry is never a change, a confirmed change still is, and the bhavcopy loader
honours an end date. Written by an AI agent (Agent A of the backtest round)."""

import importlib.util
from pathlib import Path

import pandas as pd

_PATH = Path(__file__).resolve().parents[1] / "experiments" / "expiry_event_study.py"
_spec = importlib.util.spec_from_file_location("expiry_event_study_fix", _PATH)
es = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(es)


def _table(dates, symbol="BANKNIFTY", kind="monthly"):
    d = pd.to_datetime(dates)
    return pd.DataFrame({"symbol": symbol, "expiry": d, "weekday": [es.WD[x.weekday()] for x in d],
                         "kind": kind, "source": "bhavcopy"})


def test_sample_opening_with_holiday_shifts_seeds_no_false_regime():
    # Bank Nifty monthlies as they appear when the data start in January 2023: 25 Jan and 29 Mar are
    # Wednesdays because 26 Jan and 30 Mar were holidays; the schedule is Thursday throughout.
    dates = ["2023-01-25", "2023-02-23", "2023-03-29", "2023-04-27", "2023-05-25", "2023-06-28",
             "2023-07-27", "2023-08-31", "2023-09-28", "2023-10-26", "2023-11-30", "2023-12-28"]
    ch, fl = es.detect_schedule_changes(_table(dates), data_end=pd.Timestamp("2024-01-15"))
    assert len(ch) == 0
    assert set(fl["scheduled_weekday"]) == {"Thu"}
    assert fl.loc[fl["shifted"], "expiry"].dt.strftime("%Y-%m-%d").tolist() == ["2023-01-25", "2023-03-29", "2023-06-28"]


def test_same_result_when_sample_starts_with_confirmed_thursdays():
    # the committed data start in October 2022 with three Thursdays; the fix must not change that case
    dates = ["2022-10-27", "2022-11-24", "2022-12-29", "2023-01-25", "2023-02-23", "2023-03-29", "2023-04-27"]
    ch, fl = es.detect_schedule_changes(_table(dates), data_end=pd.Timestamp("2023-05-15"))
    assert len(ch) == 0
    assert fl["scheduled_weekday"].eq("Thu").all()
    assert fl["shifted"].sum() == 2


def test_isolated_monday_before_a_tuesday_holiday_is_a_shift_not_a_change():
    # Tuesday regime; 23 Nov 2026 is a Monday because Tuesday 24 Nov is a holiday
    dates = ["2026-06-30", "2026-07-28", "2026-08-25", "2026-09-29", "2026-10-27", "2026-11-23", "2026-12-29"]
    ch, fl = es.detect_schedule_changes(_table(dates), data_end=pd.Timestamp("2027-01-10"))
    assert len(ch) == 0
    assert fl.loc[fl["expiry"] == "2026-11-23", "shifted"].item()
    assert fl["scheduled_weekday"].eq("Tue").all()


def test_confirmed_change_is_still_detected_once():
    # two Thursdays, then five Wednesdays: a change, not a run of shifts (Bank Nifty monthly, March 2024)
    dates = ["2024-01-25", "2024-02-29", "2024-03-27", "2024-04-24", "2024-05-29", "2024-06-26", "2024-07-31"]
    ch, _ = es.detect_schedule_changes(_table(dates), data_end=pd.Timestamp("2024-08-15"))
    assert len(ch) == 1
    r = ch.iloc[0]
    assert (r["change"], r["old_weekday"], r["new_weekday"]) == ("weekday", "Thu", "Wed")
    assert pd.Timestamp(r["first_new_expiry"]) == pd.Timestamp("2024-03-27")


def test_early_change_to_a_later_weekday_is_detected():
    # Wednesday regime for two expiries, then Thursdays: the detector must not read the Wednesdays as shifts
    dates = ["2023-01-25", "2023-02-22", "2023-03-30", "2023-04-27", "2023-05-25", "2023-06-29"]
    ch, _ = es.detect_schedule_changes(_table(dates), data_end=pd.Timestamp("2023-07-15"))
    assert len(ch) == 1 and (ch.iloc[0]["old_weekday"], ch.iloc[0]["new_weekday"]) == ("Wed", "Thu")


def test_bhavcopy_files_honour_end(tmp_path):
    for d in ("20261030", "20261102", "20261103"):
        (tmp_path / f"fo_{d}.csv.gz").write_bytes(b"")
    assert [f.name for f in es.bhavcopy_files([tmp_path])] == ["fo_20261030.csv.gz", "fo_20261102.csv.gz", "fo_20261103.csv.gz"]
    assert [f.name for f in es.bhavcopy_files([tmp_path], "2026-11-02")] == ["fo_20261030.csv.gz", "fo_20261102.csv.gz"]
