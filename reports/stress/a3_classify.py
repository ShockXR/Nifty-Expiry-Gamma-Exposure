"""Assign the failure class to each raw case of a3_raw_cases.csv (reviewed by hand against the observed column).
DETECTED = engine raises its own error or returns NaN that the caller can see. SAFELY HANDLED = sensible, correct result.
SILENTLY IGNORED = accepted with no error and a wrong, misleading or degenerate number. CRASHED = loud failure from library internals."""
import pandas as pd
d = pd.read_csv("a3_raw_cases.csv", keep_default_na=False)
D_, S_, I_, C_ = "DETECTED", "SAFELY HANDLED", "SILENTLY IGNORED", "CRASHED"
K = {
 0:(S_,"control",""),
 1:(I_,"implementation gap","net GEX 0 is right for zero OI, but call wall, put wall and max pain silently return the first strike (21600)"),
 2:(I_,"implementation gap","negative open interest accepted, net GEX changes sign"),
 3:(I_,"implementation gap","NaN rows are dropped by pandas sum: net GEX -7.5e9 instead of -4.0e10, no warning"),
 4:(I_,"implementation gap","an all-NaN open interest column gives net GEX exactly 0.0 (looks like a neutral day), daily_summary also 0.0"),
 5:(I_,"implementation gap","IV solver failed everywhere gives net GEX exactly 0.0, not NaN"),
 6:(I_,"implementation gap","25 percent NaN IV rows silently dropped: -1.58e11 instead of -4.0e10"),
 7:(I_,"implementation gap","empty chain gives net GEX 0.0"),
 8:(I_,"implementation gap","T=0 gives NaN gamma, net GEX 0.0, RuntimeWarning only"),
 9:(I_,"implementation gap","T<0 same as T=0"),
 10:(S_,"model property","finite; gamma scales as 1/sqrt(T) so callers must floor T near expiry"),
 11:(S_,"model property","finite, 2.2e13 at one second to expiry; no cap in the engine"),
 12:(I_,"implementation gap","IV=0 gives NaN gamma, net GEX 0.0, RuntimeWarning only"),
 13:(I_,"implementation gap","negative IV gives negative gross GEX, no check"),
 14:(S_,"model property","500 percent IV is mathematically defined"),
 15:(I_,"implementation gap","stale IV of 0.0001 gives gross GEX 150 times larger, no IV bounds"),
 16:(I_,"implementation gap","spot=0 gives NaN, net 0.0, RuntimeWarning only"),
 17:(I_,"implementation gap","negative spot same"),
 18:(I_,"implementation gap","lot size 0 gives 0.0"),
 19:(I_,"implementation gap","negative lot size flips the sign of GEX"),
 20:(I_,"invalid assumption","open interest in shares instead of contracts inflates GEX exactly 75 times; the engine has no unit check"),
 21:(I_,"invalid assumption","lot size 1 instead of 75 shrinks GEX to 1.3 percent"),
 22:(S_,"by design",""),
 23:(S_,"by design","flip is NaN when no sign change exists"),
 24:(I_,"data problem","without strikes within 2 percent of spot the net GEX changes sign and the flip moves by 430 points, no coverage check"),
 25:(I_,"data problem","calls only: net GEX +2.0e11 instead of -4.0e10"),
 26:(I_,"data problem","duplicated rows double count open interest (net -1.05e11 instead of -4.0e10)"),
 27:(I_,"implementation gap","strings CE/PE are cast to bool, every string is True, so all options count as calls: net = gross"),
 28:(S_,"by design",""),
 29:(D_,"by design",""),30:(D_,"by design",""),
 31:(I_,"implementation gap","weights outside [-1, 1] accepted (net scaled 8 times)"),
 32:(C_,"library","KeyError from pandas, no message about the schema"),33:(C_,"library","KeyError from pandas"),
 34:(S_,"by design","coerced to float"),35:(S_,"by design",""),
 36:(I_,"implementation gap","same as row 1: arbitrary first strike as wall and max pain"),
 37:(S_,"by design","NaN"),38:(C_,"library","numpy ValueError on an empty sequence"),
 39:(S_,"by design","every OTM option solved; NaN only for ITM options with no resolvable time value; price error below 1e-11 relative"),
 40:(D_,"by design","impossible prices return NaN; a price just above discounted intrinsic returns the bracket cap 5.0 as if it were an IV"),
}
assert set(K) == set(range(len(d)))
d.insert(0, "id", [f"A3-{i+1:02d}" for i in range(len(d))])
d["class"] = [K[i][0] for i in range(len(d))]; d["root_cause"] = [K[i][1] for i in range(len(d))]; d["note"] = [K[i][2] for i in range(len(d))]
d.to_csv("a3_gex_stress_cases.csv", index=False); print(d["class"].value_counts().to_string())
