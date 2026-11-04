# Did losing the weekly expiry remove the expiry-weekday volatility premium? A synthetic control study (P96)

**REAL DATA.** NSE's daily index file (Open, High, Low and Close of every NSE index) and NSE F&O bhavcopies
(for expiry calendars only), 4 September 2023 to 29 August 2025. Every number in this document comes from a file
written by `experiments/synthetic_control.py` (folder `results/synthetic_control/`, log
`results/synthetic_control_stdout.txt`), unless it is marked "post hoc" or "from the pre-registration". The only
SYNTHETIC data are in the unit tests (`tests/test_synthetic_control.py`) and in the power simulations of the
pre-registration.

**Pre-registration.** The plan was written before any outcome was computed and committed to git as
`docs/prereg/P96_SYNTHETIC_CONTROL.md`, commit **96340548fe97f80081720f8dc28034fa0302a2ce** (3 November 2026,
01:12 UTC). The script, its tests and the plan in that commit are byte for byte the ones run here. The analysis was
run once, after that commit, with `analyse --phase2 --full data/raw/nse_full`. The downloaded data have the
pre-registered fingerprint (SHA-256 `49e64d24...c791d223`, 944 index files, 512 F&O files; `run_info.json`).
**There are no deviations from the plan** (section 13).

No finance or statistics background is assumed.

## Findings first

1. **Pre-registered verdict for Bank Nifty: no support.** Against a synthetic Bank Nifty built from 19 sector
   indices without options, Bank Nifty's Wednesday did not become calmer after its weekly options ended. The gap
   moved by **+0.12 log points** (the wrong sign for the prediction), with **placebo p = 0.95**: 18 of the 19
   placebo indices showed a change at least as unusual. The pre-registered description is **"no
   pre-period premium over the synthetic control to remove"**: before the change, Bank Nifty's Wednesday was
   already slightly *less* volatile than its synthetic control's (-0.07).
2. **The study is far too imprecise to rule out the effects that matter.** It could detect, with 80% probability,
   only a change of about **1.44 log points** (a factor of about 4 in variance). The 95% confidence set for the
   effect runs from **-0.87 to +1.11**. Every whole-day premium in the event study's regime table (none larger
   than 0.25 log points either way) sits well inside that range. So, in plain words: **too imprecise to tell.**
3. **Fin Nifty and Midcap Select (secondary): no support, and they could not have shown support.** Their gaps
   also moved the wrong way (+0.28 and +0.37), with placebo p = 0.95 and 0.60 (Holm-adjusted 1.00 for both).
   As stated in the plan, with 19 donors no secondary can reach a Holm-adjusted p below 0.10.
4. **No sensitivity analysis changes this.** Across the main run and its 8 variants for the three indices (27
   runs), no ratio p is below 0.20, and only 2 of the 27 effects are negative (Bank Nifty with the long post window, -0.03; Midcap Select with the
   classic synthetic control, -0.07), both with p of 0.90 or more.
5. **The simple cross-check points the other way, but ordinary drift is as large (part post hoc).** Compared with
   the plain average of the 19 donors, each treated index's old expiry weekday became *more* volatile relative to
   its week: +0.24 (Bank Nifty), +0.38 (Fin Nifty), +0.41 (Midcap Select). Each of these is larger than the same
   number for every one of the 19 donors (post hoc reading of the placebo ranks). But the fake date of 1 April
   2024, when nothing changed, gives effects of the same size for Fin Nifty (+0.28) and Midcap Select (+0.40),
   and the synthetic control, which compares each index with the donors that track it best, finds nothing
   unusual. This is a pattern to test on new data, not a finding.
6. **Descriptive (post hoc): Wednesdays calmed down across the market, not only for Bank Nifty.** Before
   the change, the 19 sector indices without options were, on average, more volatile on Wednesday than in the rest
   of their week (+0.28), about as much as Bank Nifty (+0.19). After the change both fell, the sector indices by
   more. This study cannot tell whether that market-wide fall was itself caused by the end of Bank Nifty's weekly
   expiry (a spillover, which the method treats as "no effect") or by something else.

## Words used in this document

* **Log variance and Parkinson variance.** For each index and day, the Parkinson (1980) variance is
  (ln(High / Low))^2 / (4 ln 2): a measure of how much the index moved during the day, built from the day's
  highest and lowest values. We use its logarithm.
* **Week-relative value.** The log variance on one weekday minus the average log variance of the other trading
  days of the same week. Anything that makes the whole week volatile cancels. A positive value means that weekday
  was more volatile than the rest of its week.
* **Log points.** Differences of logs. +0.30 means about 1.35 times; -0.30 means about 0.74 times; 1.44 means
  about 4.2 times.
* **Premium.** The average week-relative value of the expiry weekday: how much more volatile that day was than
  the rest of its week.
* **Treated index.** An index that lost its weekly options: Bank Nifty (old expiry day Wednesday), Fin Nifty
  (Tuesday), Midcap Select (Monday).
* **Donor.** An index without options used to build the comparison. There are J = 19 donors.
* **Synthetic control.** A weighted average of donors, with weights chosen so that before the change it moved
  like the treated index from week to week (Abadie, Diamond and Hainmueller 2010).
* **Gap and effect.** The gap is the treated index minus its synthetic control, week by week, after each series
  has had its own pre-period average removed. The effect is the average gap after the change: how much the
  treated index's premium changed, minus how much the synthetic control's changed.
* **RMSPE.** Root mean squared gap: the typical size of the gap. "Pre RMSPE" measures how well the synthetic
  control fits before the change, "post RMSPE" how far apart they are after.
* **Placebo p (ratio p).** We pretend each of the 19 donors was treated and build its synthetic control from the
  other 18. For all 20 indices we compute post RMSPE divided by pre RMSPE. The p-value is the share of the 20
  whose ratio is at least the treated index's. The smallest possible value is 1/20 = 0.05.
* **Holm correction.** A rule that raises p-values when several questions are tested together.
* **MDE80.** The smallest true effect the test would detect 80% of the time.

## 1. The question

SEBI's circular of 1 October 2024 allowed weekly index options on only one benchmark index per exchange from 20
November 2024. On NSE, Nifty 50 kept its weekly options; Bank Nifty, Fin Nifty and Midcap Select lost theirs. If
an expiry day is more volatile *because* options expire on it, that extra volatility should disappear once the
weekly options are gone. The plan predicted a **negative** effect for Bank Nifty's Wednesday.

What was already known (`docs/EVENT_STUDY.md`, REAL DATA): Bank Nifty's Wednesday premium was small over the whole
day (+0.04 with 5-minute variance, +0.15 with per-minute Rogers-Satchell) and clearer in the last 30 minutes
(+0.32), and it fell after November 2024; but Nifty's own Wednesday also fell, so it was unclear how much of the
fall was specific to Bank Nifty. This study compares Bank Nifty with indices that never had options in the
window, which the November 2024 rules did not touch directly.

## 2. Data and checks

* NSE's full daily index files for all 944 trading days from 2 January 2023 to 2 November 2026, downloaded
  unmodified from NSE's archive; the repository's own copies hold only three indices. All 2,832 rows those copies
  hold are identical in the download (checked in phase 1).
* NSE F&O bhavcopies, 1 August 2023 to 29 August 2025, cut to index futures and options rows, for the expiry
  calendars of Fin Nifty and Midcap Select.
* 488 trading days in the study window (4 September 2023 to 29 August 2025), without the one-hour Diwali session
  of 1 November 2024.
* One weekly value was recomputed by hand from the raw files as a check (Bank Nifty and Nifty Media, week of 4
  September 2023): both agree with `weekly_BN.csv` to 12 decimal places.

## 3. Donors (J = 19)

From the 148 index names in the window: present on all 488 days (102), with Open, High, Low and Close on every day
and High above Low (68), sector or thematic rather than broad, strategy or non-equity (25), no listed index
derivatives (23), and no heavy overlap with a treated index by name (19). The donors are Nifty Auto, CPSE,
Commodities, Consumer Durables, Energy, FMCG, Healthcare Index, IT, India Consumption, India Digital, India
Manufacturing, Infrastructure, MNC, Media, Metal, Oil & Gas, PSE, Pharma and Realty (`run_info.json`,
`donor_table.csv`).

## 4. Outcome and weeks

The weekly outcome is the week-relative log Parkinson variance on the treated index's old expiry weekday, for the
treated index and for every donor on the same weeks. Weeks with a monthly or holiday-shifted expiry of Nifty or of
the treated index are left out, as are weeks in which the target day did not trade or fewer than 2 other days
traded.

## 5. Windows

| | Bank Nifty (Wednesday) | Fin Nifty (Tuesday) | Midcap Select (Monday) |
| --- | --- | --- | --- |
| before (fit) | 37 weeks, 4 Sep 2023 to 27 Sep 2024 | 35 weeks, same dates | 29 weeks, same dates |
| left out (circular to last weekly) | 30 Sep to 15 Nov 2024 | 30 Sep to 22 Nov 2024 | 30 Sep to 22 Nov 2024 |
| after (26 calendar weeks) | 17 weeks, 18 Nov 2024 to 16 May 2025 | 17 weeks, 25 Nov 2024 to 23 May 2025 | 15 weeks, same as Fin Nifty |

(REAL DATA, `results.csv`, `weekly_*.csv`.) The first usable week after the change is the week of 2 December 2024
for all three.

## 6. Estimator

Each series is centred on its own pre-period average; the weights (non-negative, adding to one) minimise the
squared pre-period gap; the effect is the average post-period gap (pre-registration, section 6). The weights
(REAL DATA, `weights_*.csv`):

* Bank Nifty: Consumer Durables 0.55, Media 0.21, Pharma 0.19, IT 0.05.
* Fin Nifty: Realty 0.43, India Consumption 0.28, IT 0.21, Infrastructure 0.09.
* Midcap Select: Pharma 0.45, Commodities 0.23, Realty 0.11, Metal 0.09, FMCG 0.05, Media 0.03, Auto 0.03.

The synthetic controls fit the treated indices less well than they fit the donors: Bank Nifty's pre RMSPE is 0.75
log points, against 0.28 to 0.67 for the 19 donor placebos; Fin Nifty's is 1.04, above every donor. That is
expected (the donors were chosen to exclude banks, financial companies and midcaps), and the ratio p allows for
it, but it is the main reason the test is imprecise.

## 7. Primary result: Bank Nifty's Wednesday

**Table 1. Main results (REAL DATA, `results.csv`, `decision.json`, `test_inversion_*.csv`)**

| index | weeks before/after | effect | 95% confidence set | ratio p (rank of 20) | signed p | premium over synthetic before | pre RMSPE | post RMSPE | MDE80 (log points) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Bank Nifty (Wed), primary** | 37/17 | **+0.12** | **-0.87 to +1.11** | **0.95 (19)** | 0.60 | **-0.07** | 0.75 | 0.58 | **1.44** |
| Fin Nifty (Tue) | 35/17 | +0.28 | -1.66 to +2.23 | 0.95 (19) | 0.75 | -0.10 | 1.04 | 0.75 | 2.03 |
| Midcap Select (Mon) | 29/15 | +0.37 | -0.51 to +1.24 | 0.60 (12) | 0.95 | -0.46 | 0.65 | 0.69 | 1.46 |

"Rank of 20" is the treated index's place when the 20 ratios are sorted from largest to smallest (1 would be the
most unusual). "Signed p" is the one-sided rank of the standardized effect (effect divided by pre RMSPE) for a
*fall*; it is reported but was not the decision statistic. All three confidence sets are one unbroken interval
and none reaches the end of the grid (-3 to +3).

**How to read it.** Bank Nifty's Wednesday gap went up a little (+0.12) rather than down, and its post/pre ratio
(0.77) is the second smallest of the 20 indices: after the change Bank Nifty tracked its synthetic control
*better* than before, not worse. There is no sign of an effect, in either direction.

**Pre-registered verdict (section 7 of the plan):** p = 0.95 is above 0.05, so the verdict is "no support". Of the
three descriptions, the first that applies is **"no pre-period premium over the synthetic control to remove"**,
because Bank Nifty's Wednesday was 0.07 log points *less* volatile than its synthetic control's before the change.
The placebo in time (fake date 1 April 2024) gives an effect of -0.01 and p = 0.95, so the method did not produce a
false "effect" in a period with no change.

## 8. Secondaries: Fin Nifty (Tuesday) and Midcap Select (Monday)

Both moved the wrong way for the prediction (Table 1). Raw ratio p: Fin Nifty 0.95, Midcap Select 0.60;
Holm-adjusted: 1.00 and 1.00. The smallest Holm-adjusted p this design allows is 0.10, so neither could have
reached 0.05 (stated in the plan). Pre-registered verdict for both: **"no support (too imprecise to tell)"**.

**Joint test of all three (supplementary, not a decision input).** The average standardized effect of the three
is +0.33. Its one-sided p for a fall, against all 6,859 combinations of donor placebos, is **0.97**. Read the
other way (post hoc), about 3.4% of the combinations have an average at least as high, so the three positive
effects together are somewhat unusual; section 9 explains why this should not be read as evidence of a rise.

## 9. Placebos, confidence sets and cross-checks

**Table 2. Placebo in time and the difference-in-differences cross-check (REAL DATA, `results.csv`)**

| index | placebo in time (1 Apr 2024): effect, ratio p | DiD against the donor average | its HAC p (two-sided) | registered rank p (one-sided, fall) |
| --- | --- | --- | --- | --- |
| Bank Nifty | -0.01, 0.95 | +0.24 | 0.157 | 1.00 |
| Fin Nifty | +0.28, 0.75 | +0.38 | 0.121 | 1.00 |
| Midcap Select | +0.40, 0.10 | +0.41 | 0.017 | 1.00 |
| Nifty 50 Wednesday (descriptive) | not run | -0.19 | 0.104 | 0.10 |

* **The difference-in-differences (DiD)** compares each treated index with the plain average of all 19 donors.
  All three rose. The registered rank p is for a *fall*, and 1.00 means that every donor's own DiD is below the
  treated index's: read the other way (post hoc), each treated index has the largest rise of the 20 (upper-tail
  p = 0.05 for each). Midcap Select's HAC p is 0.017, and it stays below 0.05 in all 8 of its variants (0.006 to
  0.044); elsewhere only Bank Nifty S8 (0.013) and Fin Nifty S4 (0.009) are below 0.05.
* **Why this is not evidence of a rise.** (1) The placebo in time, where nothing changed, gives Fin Nifty and
  Midcap Select effects (+0.28, +0.40) and DiDs (+0.28, +0.38) as large as the real ones: weekday premia of these
  indices drift by this much over six months anyway. The event study found the same: HAC p-values are too
  optimistic for weekday premia because of such drift. (2) The treated series are noisier than any donor's (pre
  RMSPE above), so their DiDs spread wider in both directions; on the standardized scale the synthetic control
  finds nothing unusual (signed p 0.60, 0.75, 0.95). (3) The plain donor average does not track the treated
  indices; the synthetic control does, a little better, and its ratio test is the registered one.
* **Nifty 50's Wednesday** (descriptive, as a fake treated unit): -0.19, ratio p 0.60, signed p 0.10. Nifty kept
  its weekly options on Thursday; its Wednesday moved down relative to the donors, but not unusually so.

## 10. Sensitivity analyses

**Table 3. Effect and ratio p for every registered variant (REAL DATA, `results.csv`; weeks before/after in
brackets)**

| run | Bank Nifty | Fin Nifty | Midcap Select |
| --- | --- | --- | --- |
| main | +0.12, 0.95 (37/17) | +0.28, 0.95 (35/17) | +0.37, 0.60 (29/15) |
| S1 long post window, to 29 Aug 2025 | -0.03, 1.00 (37/28) | +0.23, 0.95 (35/27) | +0.42, 0.85 (29/25) |
| S2 Garman-Klass variance | +0.16, 0.95 (37/17) | +0.27, 0.90 (35/17) | +0.30, 0.85 (29/15) |
| S3 classic synthetic control (no centring) | +0.05, 0.90 (37/17) | +0.18, 0.95 (35/17) | -0.07, 0.90 (29/15) |
| S4 pre-period ends 26 Jul 2024 | +0.09, 0.95 (32/17) | +0.57, 0.55 (30/17) | +0.44, 0.40 (24/15) |
| S5 baseline without Nifty and Sensex expiry days | +0.09, 0.80 (32/14) | +0.33, 0.80 (30/16) | +0.36, 0.85 (25/14) |
| S6 without the largest-weight donor | +0.25, 0.95 (37/17) | +0.24, 0.95 (35/17) | +0.34, 0.63 (29/15) |
| S7 at least 3 other days | +0.14, 0.95 (37/16) | +0.36, 0.95 (35/16) | +0.37, 0.60 (29/15) |
| S8 baseline Thursday and Friday only | +0.23, 0.95 (36/15) | +0.42, 1.00 (34/15) | +0.41, 0.20 (28/14) |

S6 drops Consumer Durables (Bank Nifty), Realty (Fin Nifty) and Pharma (Midcap Select); with 18 donors the
smallest p is 1/19. No variant comes near p = 0.05, and 25 of the 27 effects are positive. S8, which removes the
other treated indices' old expiry days from the baseline, makes every effect larger, not smaller.

## 11. Power: what this study could and could not detect

* **The smallest possible p is 1/20 = 0.05.** The primary test can reject only if Bank Nifty's ratio is the
  largest of all 20 indices. For the secondaries, after Holm, the floor is 0.10.
* **MDE80 of the registered test, in log points** (pre-registered multiplier times the real pre RMSPE): Bank
  Nifty **1.44** (a factor of about 4.2 in variance), Fin Nifty **2.03** (about 7.6), Midcap Select **1.46**
  (about 4.3). The signed test, which was not the decision statistic, would have needed about half as much
  (0.79, 1.14 and 0.82 log points; same arithmetic with the plan's multipliers for that test).
* **The confidence sets say the same.** For Bank Nifty the data are consistent with any constant effect from
  -0.87 to +1.11 log points, that is, from a fall to about 0.42 times to a rise to about 3 times.
* **Compared with what was being looked for.** Bank Nifty's own Wednesday premium before the change was +0.19
  with this measure (post hoc, `weekly_BN.csv`) and +0.15 with per-minute Rogers-Satchell in the event study.
  Removing a premium of that size is a change of about one eighth of the MDE80 or less. The study could not have detected
  it. **A null result here means "too imprecise to tell", not "no effect".**

## 12. What this means

* **For the question asked:** this design gives no evidence that losing the weekly expiry removed a volatility
  premium on the old expiry weekday of Bank Nifty, Fin Nifty or Midcap Select. It also cannot rule out that it
  did, for any premium of a realistic size.
* **Descriptive (post hoc, from `weekly_*.csv`):** the average week-relative value before and after the change
  (Table 4).
* **A pattern for a new pre-registered test (post hoc):** all three treated indices' old expiry weekdays became
  relatively *more* volatile than the plain donor average after the change, the opposite of the prediction. The
  placebo in time shows drift of the same size, so it should be tested on new data, with a design that has more
  power, before anyone reads it as a result.

**Table 4. Average week-relative value of the old expiry weekday, before and after (REAL DATA, post hoc,
`weekly_*.csv`)**

| | treated index: before, after | its synthetic control: before, after | average of the 19 donors: before, after |
| --- | --- | --- | --- |
| Bank Nifty, Wednesday | +0.19, -0.02 | +0.26, -0.07 | +0.28, -0.17 |
| Fin Nifty, Tuesday | -0.17, +0.24 | -0.07, +0.06 | +0.02, +0.05 |
| Midcap Select, Monday | -0.65, +0.01 | -0.18, +0.10 | -0.21, +0.03 |

Before the change, none of the three treated indices was more volatile on its own expiry day than the comparison
indices were on the same weekday; Midcap Select's Monday was much calmer. Wednesdays were volatile across the
market before November 2024 and calmer after, for indices with and without options. If the end of Bank Nifty's
weekly expiry calmed every Wednesday through hedging spillovers, this method would show no effect by construction
(the plan says so in its section 6); the data here cannot separate that from other reasons.

## 13. Deviations from the pre-registration

**None.** The registered script ran once, unchanged, on data with the registered fingerprint; no bug was found and
no code was changed. For transparency:

* Added after seeing the results and labelled post hoc wherever used: the before and after averages in section 12,
  the upper-tail readings of the DiD ranks and of the joint test (sections 8 and 9), and the hand recomputation of
  one weekly value (a check, section 2). None of them changes a verdict.
* The MDE80 values for the signed test (section 11) are arithmetic with the plan's multipliers, not new runs.
* The figure `fig_gaps.png` has overlapping date labels on its horizontal axes; it was left as produced rather than
  changing the registered script for a cosmetic fault.

## 14. Limitations

1. **Low power** (section 11). This is the main limitation, and it was stated in the plan before any data were
   seen.
2. **Spillovers.** The donors have no options, but their stocks trade in the same market. If an expiry's effect
   spread to the whole market, the synthetic control absorbs it and the estimate understates the effect.
3. **One measure from two prices a day.** The Parkinson variance uses only the day's high and low, which is noisy;
   the end-of-day premium found in the event study (last 30 minutes) cannot be isolated with daily data.
4. **Donor overlap is judged by name**, because historical constituent lists are not available; dropping the
   largest-weight donor (S6) does not change anything.
5. **The November 2024 package bundled several changes** (larger contracts and extra margin on expiry day, per
   `docs/EVENT_STUDY.md`; and, as reported but not verified, the end of BSE's Bankex and Sensex 50 weeklies and
   Sensex's move from Friday to Tuesday in January 2025). The week-relative
   design removes changes that hit all weekdays equally, but not changes that hit one weekday.
6. **Bank Nifty's own series was partly known** from the event study before this plan was written (other
   measures); the donor series and gaps were not.

## 15. Files and how to rerun

* Plan: `docs/prereg/P96_SYNTHETIC_CONTROL.md`. Script: `experiments/synthetic_control.py`. Tests (SYNTHETIC
  data only): `tests/test_synthetic_control.py`.
* Outputs in `results/synthetic_control/`: `results.csv` (one row per run), `decision.json` (verdicts, Holm, joint
  test), `placebo_space_{BN,FIN,MID}.csv`, `test_inversion_{BN,FIN,MID}.csv` (confidence sets),
  `weekly_{BN,FIN,MID}.csv` (every weekly value, with the gap), `weights_{BN,FIN,MID}.csv`, `donor_table.csv`,
  `fig_gaps.png` (the gap of each treated index in colour, the 19 placebo gaps in grey; dashed line at 20
  November 2024), `run_info.json` (donors, data fingerprint, library versions). Log:
  `results/synthetic_control_stdout.txt`.

```bash
OMP_NUM_THREADS=1 python experiments/synthetic_control.py fetch --out data/raw/nse_full    # only if the files are missing
OMP_NUM_THREADS=1 python experiments/synthetic_control.py analyse --phase2 --full data/raw/nse_full   # about 1 minute
OMP_NUM_THREADS=1 python -m pytest -q tests/test_synthetic_control.py
```

The analysis has no random parts, so a rerun on the same files gives the same numbers.

## References

Abadie, A., Diamond, A. and Hainmueller, J. (2010), "Synthetic Control Methods for Comparative Case Studies",
Journal of the American Statistical Association 105(490), 493-505. Parkinson, M. (1980), "The Extreme Value Method
for Estimating the Variance of the Rate of Return", Journal of Business 53(1), 61-65. Further references, all cited
from memory, are listed in the pre-registration.
