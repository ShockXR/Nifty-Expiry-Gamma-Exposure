"""Nifty weekly expiry calendar and regime labels.

Scheduled Nifty weekly expiry weekday (configurable, check against NSE
circulars before relying on it):

* Thursday, up to and including 31 August 2025.
* Tuesday, from 1 September 2025.

If the scheduled day is a holiday, the contract expires on the previous
trading day. Rather than hard-coding a holiday list, this module takes the
set of trading days that actually appear in the data and moves each
scheduled expiry back to the nearest trading day on or before it.

Regimes used in the research design:

* ``multi_weekly``      before 20 Nov 2024 (several benchmark weeklies per exchange)
* ``single_weekly_thu`` 20 Nov 2024 to 31 Aug 2025 (SEBI one weekly benchmark expiry per exchange)
* ``single_weekly_tue`` from 1 Sep 2025 (Nifty weekly moved to Tuesday)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ExpiryRules:
    switch_to_tuesday: pd.Timestamp = pd.Timestamp("2025-09-01")
    single_weekly_start: pd.Timestamp = pd.Timestamp("2024-11-20")
    weekday_before: int = 3  # Thursday
    weekday_after: int = 1  # Tuesday
    regimes: list = field(
        default_factory=lambda: [
            (pd.Timestamp("1900-01-01"), "multi_weekly"),
            (pd.Timestamp("2024-11-20"), "single_weekly_thu"),
            (pd.Timestamp("2025-09-01"), "single_weekly_tue"),
        ]
    )

    def scheduled_weekday(self, day: pd.Timestamp) -> int:
        return self.weekday_after if day >= self.switch_to_tuesday else self.weekday_before

    def regime(self, day: pd.Timestamp) -> str:
        label = self.regimes[0][1]
        for start, name in self.regimes:
            if day >= start:
                label = name
        return label


def weekly_expiries(trading_days, rules: ExpiryRules | None = None) -> pd.DatetimeIndex:
    """Actual weekly expiry dates given the trading days present in the data."""
    rules = rules or ExpiryRules()
    days = pd.DatetimeIndex(sorted(pd.to_datetime(trading_days))).normalize()
    if len(days) == 0:
        return days
    out = []
    # Walk calendar weeks; find the scheduled weekday, then roll back to a trading day.
    start = days[0] - pd.Timedelta(days=days[0].weekday())
    week = start
    day_set = set(days)
    while week <= days[-1] + pd.Timedelta(days=7):
        # Use the rule in force on the Monday of that week.
        wd = rules.scheduled_weekday(week)
        sched = week + pd.Timedelta(days=wd)
        d = sched
        while d not in day_set and d > week - pd.Timedelta(days=1):
            d -= pd.Timedelta(days=1)
        if d in day_set and days[0] <= d <= days[-1]:
            out.append(d)
        week += pd.Timedelta(days=7)
    return pd.DatetimeIndex(sorted(set(out)))


def label_days(trading_days, rules: ExpiryRules | None = None, expiries=None) -> pd.DataFrame:
    """One row per trading day with expiry flags, days to next expiry and regime.

    ``expiries``: optional list of actual expiry dates (for example read from
    the bhavcopies with data.actual_expiries). If given, it replaces the
    weekday rule; only dates that are trading days in ``trading_days`` count.
    """
    rules = rules or ExpiryRules()
    days = pd.DatetimeIndex(sorted(pd.to_datetime(trading_days))).normalize()
    if expiries is None:
        exp = weekly_expiries(days, rules)
    else:
        exp = pd.DatetimeIndex(sorted(set(pd.DatetimeIndex(expiries).normalize()) & set(days)))
    exp_arr = exp.values
    pos = np.searchsorted(exp_arr, days.values)
    next_exp = pd.DatetimeIndex([exp_arr[p] if p < len(exp_arr) else pd.NaT for p in pos])
    day_idx = {d: i for i, d in enumerate(days)}
    tdte = [day_idx[n] - day_idx[d] if pd.notna(n) else np.nan for d, n in zip(days, next_exp)]
    monthly = set()
    s = pd.Series(exp, index=exp)
    for _, grp in s.groupby([exp.year, exp.month]):
        monthly.add(grp.iloc[-1])
    df = pd.DataFrame(
        {
            "date": days,
            "is_expiry": days.isin(exp),
            "is_monthly_expiry": days.isin(list(monthly)),
            "next_expiry": next_exp,
            "trading_days_to_expiry": tdte,
            "regime": [rules.regime(d) for d in days],
            "weekday": days.weekday,
        }
    )
    return df.set_index("date")
