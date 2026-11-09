"""
02b_extensive_margin.py
=======================
Why does OLS on positive flows (spec A2) give a much smaller language effect
than PPML (spec A1)?  OLS on log flows throws away every pair with zero
commuters. PPML keeps them. So if the language border mainly decides WHETHER
anyone commutes between two places at all, OLS cannot see it.

Test: a linear probability model of "at least 5 commuters" (the publication
threshold) on the same controls, for pairs within 60 minutes (where commuting
is realistic).  Output appended to output/results.csv as spec E.
"""
from pathlib import Path

import pandas as pd
import pyfixest as pf

ROOT = Path(__file__).resolve().parents[1]
d = pd.read_parquet(ROOT / "data" / "clean" / "pairs.parquet",
                    filters=[("year", "==", 2018)])
d = d[d.time <= 60].copy()
d["any_flow"] = (d.flow > 0).astype(float) * 100   # in percentage points
m = pf.feols("any_flow ~ ln_time + diff_canton + diff_lang | id_o + id_d",
             data=d, vcov={"CRV1": "id_o"})
t = m.tidy()
print(t[["Estimate", "Std. Error"]].round(3))
print("baseline share of pairs (<=60 min) with a flow:", round(d.any_flow.mean(), 2), "%")

res = pd.read_csv(ROOT / "output" / "results.csv")
res = res[res.spec != "E Any commuting link (LPM, pp, <=60 min)"]
new = pd.DataFrame([dict(spec="E Any commuting link (LPM, pp, <=60 min)", method="ols",
                         note=f"baseline {d.any_flow.mean():.2f}% of pairs", var=v,
                         coef=r["Estimate"], se=r["Std. Error"], lo=r["2.5%"], hi=r["97.5%"],
                         n=m._N) for v, r in t.iterrows()])
pd.concat([res, new]).to_csv(ROOT / "output" / "results.csv", index=False)
