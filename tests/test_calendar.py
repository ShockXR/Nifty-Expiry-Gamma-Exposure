import pandas as pd

from niftygex.expiry_calendar import label_days, weekly_expiries


def test_thursday_then_tuesday_and_holiday_rollback():
    days = pd.bdate_range("2025-08-18", "2025-09-12")
    days = days.drop(pd.Timestamp("2025-08-28"))  # pretend Thursday 28 Aug is a holiday
    exp = weekly_expiries(days)
    assert pd.Timestamp("2025-08-21") in exp  # Thursday
    assert pd.Timestamp("2025-08-27") in exp  # rolled back from the holiday
    assert pd.Timestamp("2025-09-02") in exp  # first Tuesday expiry
    assert pd.Timestamp("2025-09-09") in exp


def test_labels_and_regimes():
    lab = label_days(pd.bdate_range("2024-11-11", "2024-11-29"))
    assert lab.loc["2024-11-14", "is_expiry"]
    assert lab.loc["2024-11-19", "regime"] == "multi_weekly"
    assert lab.loc["2024-11-20", "regime"] == "single_weekly_thu"
    assert lab.loc["2024-11-18", "trading_days_to_expiry"] == 3
    assert lab.loc["2024-11-28", "is_monthly_expiry"]


def test_label_days_uses_given_monthly_expiries():
    import pandas as pd
    from niftygex.expiry_calendar import label_days
    days = pd.bdate_range("2025-04-01", "2025-05-02")
    exps = pd.to_datetime(["2025-04-03", "2025-04-09", "2025-04-17", "2025-04-24", "2025-04-30"])
    lab = label_days(days, expiries=exps)
    assert lab.loc["2025-04-30", "is_monthly_expiry"]  # the calendar-month rule picks the last one
    lab = label_days(days, expiries=exps, monthly_expiries=pd.to_datetime(["2025-04-24"]))
    assert lab.loc["2025-04-24", "is_monthly_expiry"] and not lab.loc["2025-04-30", "is_monthly_expiry"]
