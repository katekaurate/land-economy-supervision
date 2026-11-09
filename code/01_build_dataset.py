"""
01_build_dataset.py
===================
Builds the commune-pair dataset used in the gravity regressions.

Question: Which border matters more for commuting in Switzerland -
          an institutional border (canton) or a cultural one (language)?

Inputs (all in data/raw/, see README for download links)
-------------------------------------------------------
1. ts-x-11.04.04.05.csv
   FSO: Employed persons by commune of residence and commune of work,
   2014, 2018, 2020. Built from linked registers (AHV/OASI, STATPOP, STATENT).
   Columns: PERSPECTIVE (R/W), REF_YEAR, GEO_CANT_RESID, GEO_COMM_RESID,
            GEO_CANT_WORK, GEO_COMM_WORK, VALUE.
   - Flows below 5 people are suppressed for confidentiality and lumped
     into a catch-all code 7777. We therefore treat any pair not shown as 0.
   - The table is published twice: once from the residence side (R) and once
     from the workplace side (W). Suppression differs slightly between the
     two, so we take the union of the two (max of the two values).
   - Communes are on (almost) 2020 boundaries in all three years.

2. DWV_2017_Strasse_Reisezeit_CH_2337.mtx
   Federal Office for Spatial Development (ARE): road travel time in minutes
   between every pair of communes, 2017 boundaries (national passenger
   transport model, NPVM 2017). Also gives own-commune travel time.

3. Commune and district boundaries: npm package `swiss-maps` v4.7.0
   (interactivethings), which redistributes the swisstopo/FSO boundary
   files. Used for (a) the map, (b) assigning each commune to a district,
   (c) linking 2020 communes to 2017 travel-time zones after mergers.

4. Raumgliederungen.xlsx
   FSO commune register tool ("Applikation der Schweizer Gemeinden"),
   spatial division "Sprachgebiete 2020": the OFFICIAL language area of every
   commune (German, French, Italian, Romansh). THIS is the language used in
   the analysis.

Language classification
-----------------------
The analysis uses the official FSO language areas (input 4). For comparison
the script also builds our own rule-based version (column lang_rule), which
agrees with the official one for 2,191 of 2,198 communes (Romansh aside).
The rule: most cantons are monolingual, so the canton decides.
In the four bilingual cantons we assign by district, with named exceptions:
  * Bern:       Jura bernois district -> French, rest -> German
                (Biel/Bienne is officially bilingual; majority German)
  * Fribourg:   Sense -> German; See/Lac -> German except four
                French-majority communes; other districts -> French,
                except Jaun (German-speaking, in Gruyere)
  * Valais:     Goms, Brig, Visp, Raron, Leuk (Upper Valais) -> German,
                rest -> French
  * Graubuenden: Moesa, Bernina and Bregaglia -> Italian; the rest -> German.
                SIMPLIFICATION: Romansh communes are grouped with German
                (almost all Romansh speakers are bilingual in German).
                Robustness check drops Graubuenden entirely.
  * Jura: Ederswiler is German-speaking.  Ticino: Bosco/Gurin is German-speaking.
Communes in the See/Lac and Biel/Bienne districts are flagged as
"mixed" so they can be dropped in a robustness check.

Output: data/clean/pairs.parquet  (one row per origin x destination x year)
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MAPS = ROOT / "data" / "swiss-maps"  # unpacked npm package swiss-maps@4.7.0 (see README)
OUT = ROOT / "data" / "clean"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Commuting flows
# ---------------------------------------------------------------------------
raw = pd.read_csv(RAW / "ts-x-11.04.04.05.csv", encoding="utf-8-sig")
raw = raw[(raw.GEO_COMM_RESID != 7777) & (raw.GEO_COMM_WORK != 7777)]

# Union of the residence-side (R) and workplace-side (W) tables
flows = (
    raw.groupby(["REF_YEAR", "GEO_COMM_RESID", "GEO_COMM_WORK"], as_index=False)
    .VALUE.max()
    .rename(columns={"REF_YEAR": "year", "GEO_COMM_RESID": "o",
                     "GEO_COMM_WORK": "d", "VALUE": "flow"})
)
canton_of = (
    pd.concat([
        raw[["GEO_COMM_RESID", "GEO_CANT_RESID"]].set_axis(["id", "canton"], axis=1),
        raw[["GEO_COMM_WORK", "GEO_CANT_WORK"]].set_axis(["id", "canton"], axis=1),
    ]).drop_duplicates("id").set_index("id").canton
)

# ---------------------------------------------------------------------------
# 2. Commune geography (2020 boundaries + one 2021 merger: 5399 Riviera TI)
# ---------------------------------------------------------------------------
def read(year, layer):
    """Read a swiss-maps layer and put it in the Swiss LV95 grid (EPSG:2056).
    Older vintages are stored in LV03 (EPSG:21781), so we reproject."""
    g = gpd.read_file(MAPS / str(year) / f"{layer}.shp", encoding="utf-8")
    g = g.to_crs(2056) if g.crs.to_epsg() == 21781 else g.set_crs(2056, allow_override=True)
    g["id"] = g["id"].astype(int)
    return g

com = read(2020, "municipalities")
ids = set(canton_of.index)
com = com[com.id.isin(ids)]
missing = ids - set(com.id)
if missing:
    com = pd.concat([com, read(2021, "municipalities").query("id in @missing")])
com = gpd.GeoDataFrame(com, crs=2056)
assert set(com.id) == ids, "some communes have no boundary"

# District of each commune (spatial join on an interior point)
dist = read(2020, "districts")[["id", "name", "geometry"]].rename(
    columns={"id": "district_id", "name": "district"})
pts = com[["id", "geometry"]].copy()
pts["geometry"] = pts.representative_point()
com = com.merge(gpd.sjoin(pts, dist, predicate="within")[["id", "district_id", "district"]],
                on="id", how="left")
com["canton"] = com.id.map(canton_of)

# ---------------------------------------------------------------------------
# 3. Language of each commune
# ---------------------------------------------------------------------------
GERMAN_CANTONS = {1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20}
FRENCH_CANTONS = {22, 24, 25, 26}
SEE_FRENCH = {"Courtepin", "Misery-Courtion", "Mont-Vully", "Courgevaux"}
UPPER_VALAIS = {2301, 2304, 2306, 2309, 2313}  # Brig, Goms, Leuk, Raron, Visp
GR_ITALIAN = {1842, 1847}                      # Bernina, Moesa


def language(row):
    c, dct, name = row.canton, row.district_id, row["name"]
    if name in {"Ederswiler", "Bosco/Gurin", "Jaun"}:
        return "German"
    if c in GERMAN_CANTONS:
        return "German"
    if c in FRENCH_CANTONS:
        return "French"
    if c == 21:
        return "Italian"
    if c == 2:   # Bern
        return "French" if dct == 241 else "German"
    if c == 10:  # Fribourg
        if dct == 1006:
            return "German"
        if dct == 1005:
            return "French" if name in SEE_FRENCH else "German"
        return "French"
    if c == 23:  # Valais
        return "German" if dct in UPPER_VALAIS else "French"
    if c == 18:  # Graubuenden
        return "Italian" if (dct in GR_ITALIAN or name == "Bregaglia") else "German"
    raise ValueError(row)


com["lang_rule"] = com.apply(language, axis=1)   # our own rule (kept for comparison)

# OFFICIAL language area (FSO "Sprachgebiete 2020", exported from the FSO
# commune register tool, data/raw/Raumgliederungen.xlsx). The export is on
# 2026 commune boundaries, so a 2020 commune that has since merged takes the
# language of its 2026 successor (the commune containing its interior point).
# Romansh is kept as its own language here.
off = pd.read_excel(RAW / "Raumgliederungen.xlsx", sheet_name="Daten",
                    header=None, skiprows=3).iloc[:, [0, 6]].dropna()
off.columns = ["key", "code"]
off["key"] = off.key.astype(int)
off["lang"] = off.code.astype(int).map({1: "German", 2: "French", 3: "Italian", 4: "Romansh"})
m26 = read(2026, "municipalities")[["id", "geometry"]].rename(columns={"id": "succ"})
pts = com[["id", "geometry"]].copy()
pts["geometry"] = pts.representative_point()
succ = gpd.sjoin(pts, m26, predicate="within").set_index("id").succ
com["key"] = com.id.where(com.id.isin(off.key), com.id.map(succ))
com = com.merge(off[["key", "lang"]], on="key", how="left").drop(columns="key")
assert com.lang.notna().all()
print("rule vs official language agree for",
      (com.lang_rule == com.lang).sum(), "of", len(com), "communes")
com["mixed_area"] = com.district_id.isin([1005, 242])  # See/Lac, Biel/Bienne
com["area_km2"] = com.area / 1e6

# ---------------------------------------------------------------------------
# 4. Travel times -> 2020 communes
#    The ARE matrix (published 2017) uses commune boundaries as of 2015.
#    For communes created by a merger after 2015 we use the predecessor
#    commune that covers the largest part of its area. Three uninhabited
#    "communes" (state forest / common land: 2391, 5391, 5394) have no
#    travel-time zone and are dropped.
# ---------------------------------------------------------------------------
tt = pd.read_csv(RAW / "DWV_2017_Strasse_Reisezeit_CH_2337.mtx", skiprows=8,
                 nrows=2337 ** 2, sep=r"\s+", header=None, names=["o", "d", "t"],
                 encoding="latin-1")
com15 = read(2015, "municipalities")[["id", "geometry"]].rename(columns={"id": "id17"})
com15 = com15[com15.id17.isin(set(tt.o))]
com = com[~com.id.isin([2391, 5391, 5394])]
ov = gpd.overlay(com[["id", "geometry"]], com15, how="intersection")
ov["a"] = ov.area
link = ov.sort_values("a").drop_duplicates("id", keep="last").set_index("id").id17
com["id17"] = com.id.map(link)
assert com.id17.notna().all()

tt_wide = tt.set_index(["o", "d"]).t
print("communes:", len(com), "| languages:", com.lang.value_counts().to_dict())

# ---------------------------------------------------------------------------
# 5. Full origin x destination grid (zeros included), per year
# ---------------------------------------------------------------------------
c = com[["id", "id17", "canton", "lang", "mixed_area", "district_id"]]
grid = c.merge(c, how="cross", suffixes=("_o", "_d"))
grid = grid[grid.id_o != grid.id_d]          # drop own-commune commuting
grid["time"] = tt_wide.reindex(
    pd.MultiIndex.from_arrays([grid.id17_o, grid.id17_d])).to_numpy()
# A few merged communes map to the same 2017 zone -> time 0; use own-zone time
same = grid.time.isna() | (grid.id17_o == grid.id17_d)
grid.loc[same, "time"] = tt_wide.reindex(
    pd.MultiIndex.from_arrays([grid.id17_o[same], grid.id17_d[same]])).to_numpy()
assert grid.time.notna().all()

grid["diff_canton"] = (grid.canton_o != grid.canton_d).astype(int)
grid["diff_lang"] = (grid.lang_o != grid.lang_d).astype(int)
grid["ln_time"] = np.log(grid.time)

panel = []
for y in sorted(flows.year.unique()):
    g = grid.merge(flows[flows.year == y][["o", "d", "flow"]],
                   left_on=["id_o", "id_d"], right_on=["o", "d"], how="left")
    g["flow"] = g.flow.fillna(0)
    g["year"] = y
    panel.append(g.drop(columns=["o", "d"]))
panel = pd.concat(panel, ignore_index=True)

keep = ["year", "id_o", "id_d", "canton_o", "canton_d", "lang_o", "lang_d",
        "district_id_o", "district_id_d", "mixed_area_o", "mixed_area_d",
        "time", "ln_time", "diff_canton", "diff_lang", "flow"]
panel[keep].to_parquet(OUT / "pairs.parquet", index=False)
com.drop(columns="geometry").to_csv(OUT / "communes.csv", index=False)
com.to_file(OUT / "communes.gpkg", driver="GPKG")

print(panel.groupby("year").agg(pairs=("flow", "size"), positive=("flow", lambda s: (s > 0).sum()),
                                commuters=("flow", "sum")))
