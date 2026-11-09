"""
02_regressions.py
=================
Gravity regressions of commuting between Swiss communes.

Model (structural gravity, estimated by PPML - Poisson pseudo-maximum likelihood):

    E[flow_ij] = exp( a_i + b_j + d*ln(time_ij) + g1*DiffCanton_ij + g2*DiffLanguage_ij )

  flow_ij       number of people living in commune i and working in commune j
  a_i, b_j      origin and destination fixed effects. These absorb everything
                about the size of i (how many workers live there) and of j
                (how many jobs are there) - the "mass" terms of the gravity
                equation in Lecture 5-6, plus Anderson-van Wincoop style
                multilateral resistance.
  ln(time_ij)   log road travel time - the "distance" term
  DiffCanton    1 if i and j are in different cantons  (institutional border)
  DiffLanguage  1 if i and j have different majority languages (cultural border)

Why PPML rather than OLS on log flows?  Most commune pairs have zero
commuters, and log(0) is undefined. PPML keeps the zeros and is the
standard estimator for gravity equations (Santos Silva & Tenreyro 2006).
We also report OLS on positive flows for comparison.

Interpretation: a coefficient g on a border dummy means crossing that border
multiplies expected commuting by exp(g). So the % reduction is 1 - exp(g).
Comparing with the time elasticity d, a border is "worth" the same as
multiplying travel time by exp(g/d).

Standard errors are clustered by origin commune.

Output: output/results.csv  (one row per coefficient per specification)
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

COLS = ["year", "id_o", "id_d", "canton_o", "canton_d", "lang_o", "lang_d",
        "district_id_o", "district_id_d", "mixed_area_o", "mixed_area_d",
        "time", "ln_time", "diff_canton", "diff_lang", "flow"]


def load(year):
    """Load one year of the pair data (all years at once does not fit in memory)."""
    d = pd.read_parquet(ROOT / "data" / "clean" / "pairs.parquet", columns=COLS,
                        filters=[("year", "==", year)])
    for c in ["ln_time", "time", "flow"]:
        d[c] = d[c].astype("float64")
    return d


VCOV = {"CRV1": "id_o"}
rows = []


def run(name, formula, data, method="ppml", note=""):
    t = time.time()
    if method == "ppml":
        m = pf.fepois(formula, data=data, vcov=VCOV)
    else:
        m = pf.feols(formula, data=data, vcov=VCOV)
    tid = m.tidy()
    for var, r in tid.iterrows():
        rows.append(dict(spec=name, method=method, note=note, var=var,
                         coef=r["Estimate"], se=r["Std. Error"],
                         lo=r["2.5%"], hi=r["97.5%"], n=m._N))
    pd.DataFrame(rows).to_csv(OUT / "results.csv", index=False)
    print(f"[{time.time()-t:5.0f}s] {name}\n{tid[['Estimate','Std. Error']].round(3)}\n", flush=True)
    del m
    import gc; gc.collect()


base = "flow ~ ln_time + diff_canton + diff_lang | id_o + id_d"
y18 = load(2018)

# --- A. Main result and robustness (2018) -----------------------------------
run("A1 Main (PPML, 2018)", base, y18)

pos = y18[y18.flow > 0].assign(ln_flow=lambda d: np.log(d.flow))
run("A2 OLS on log positive flows", "ln_flow ~ ln_time + diff_canton + diff_lang | id_o + id_d",
    pos, method="ols")

# Flexible travel-time control: dummies for 5-minute bands (up to 3 hours)
y18["time_band"] = np.minimum(y18.time // 5, 36).astype(int)
run("A3 Flexible travel-time bands", "flow ~ diff_canton + diff_lang | id_o + id_d + time_band", y18)

run("A4 Drop bilingual-area communes (See/Lac, Biel)", base,
    y18[~(y18.mixed_area_o | y18.mixed_area_d)])
run("A5 Drop Graubuenden (Romansh/Italian/German mix)", base,
    y18[(y18.canton_o != 18) & (y18.canton_d != 18)])
run("A6 Only pairs within 60 min", base, y18[y18.time <= 60])
run("A7 Interaction (both borders at once)",
    "flow ~ ln_time + diff_canton + diff_lang + diff_canton:diff_lang | id_o + id_d", y18)

# --- B. Which language pairs, and is it asymmetric? --------------------------
lo, ld = y18.lang_o, y18.lang_d
y18["de_to_fr"] = ((lo == "German") & (ld == "French")).astype(int)
y18["fr_to_de"] = ((lo == "French") & (ld == "German")).astype(int)
y18["involves_it"] = (((lo == "Italian") | (ld == "Italian")) & (lo != ld)).astype(int)
y18["involves_rm"] = (((lo == "Romansh") | (ld == "Romansh")) & (lo != ld)
                      & (lo != "Italian") & (ld != "Italian")).astype(int)
run("B1 Asymmetry: German->French vs French->German",
    "flow ~ ln_time + diff_canton + de_to_fr + fr_to_de + involves_it + involves_rm | id_o + id_d", y18,
    note="direction = home language -> work language")

# --- C. Over time -------------------------------------------------------------
for yr in [2014, 2020]:
    run(f"C {yr}", base, load(yr))

# --- D. District level (Lecture 9-10: MAUP) ---------------------------------
#  Re-run the same regression with ~140 districts instead of ~2,200 communes.
com = pd.read_csv(ROOT / "data" / "clean" / "communes.csv")
# employed residents per commune (2018) as weights for district language
res = y18.groupby("id_o").flow.sum()
com["w"] = com.id.map(res).fillna(0) + 1
dl = (com.groupby(["district_id", "lang"]).w.sum().reset_index()
      .sort_values("w").drop_duplicates("district_id", keep="last")
      .set_index("district_id").lang)
dc = com.drop_duplicates("district_id").set_index("district_id").canton
d = (y18.groupby(["district_id_o", "district_id_d"])
     .agg(flow=("flow", "sum"), time=("time", "mean")).reset_index())
d = d[d.district_id_o != d.district_id_d]
d["ln_time"] = np.log(d.time)
d["diff_canton"] = (d.district_id_o.map(dc) != d.district_id_d.map(dc)).astype(int)
d["diff_lang"] = (d.district_id_o.map(dl) != d.district_id_d.map(dl)).astype(int)
d = d.rename(columns={"district_id_o": "id_o", "district_id_d": "id_d"})
run("D District level (MAUP check)", base, d)

print("done")
