"""
03_figures.py
=============
Figure 1: map of Swiss communes coloured by majority language, with canton
          borders drawn on top - shows where the two kinds of border coincide
          and where they don't (the bilingual cantons).
Figure 2: the main result - % reduction in commuting from crossing each border.
"""
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
COL = {"German": "#2a78d6", "French": "#eb6834", "Italian": "#1baf7a"}
INK, MUTED = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

# ---------------- Figure 1: map ----------------
com = gpd.read_file(ROOT / "data" / "clean" / "communes.gpkg")
cant = com.dissolve("canton")
lakes = gpd.read_file(ROOT / "data" / "swiss-maps" / "2020" / "lakes.shp").set_crs(2056, allow_override=True)

fig, ax = plt.subplots(figsize=(10, 6.6), dpi=200)
TINT = {"German": "#9ec5f4", "French": "#f5b394", "Italian": "#8fd9bd", "Romansh": "#cbbcf2"}  # lighter steps of the same hues
for lang, c in TINT.items():
    com[com.lang == lang].plot(ax=ax, color=c, linewidth=0)
lakes.plot(ax=ax, color="#ffffff", linewidth=0)
cant.boundary.plot(ax=ax, color=INK, linewidth=0.9)
bil = cant.loc[[2, 10, 23, 18]]
bil.boundary.plot(ax=ax, color=INK, linewidth=2.2)
for k, name in {2: "Bern", 10: "Fribourg", 23: "Valais", 18: "Graubünden"}.items():
    p = cant.loc[[k]].representative_point().iloc[0]
    ax.annotate(name, (p.x, p.y), ha="center", fontsize=9, color=INK, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in TINT.values()]
handles += [plt.Line2D([], [], color=INK, lw=0.9), plt.Line2D([], [], color=INK, lw=2.2)]
ax.legend(handles, ["German-speaking", "French-speaking", "Italian-speaking", "Romansh-speaking",
                    "Canton border", "Multilingual canton"], loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False, fontsize=9)
ax.set_title("Two kinds of border: cantons (lines) and languages (colours)",
             loc="left", fontsize=13, color=INK, fontweight="bold")
ax.text(0, 1.0, "Most canton borders sit inside one language area; the language line itself "
        "runs through Bern, Fribourg and Valais.", transform=ax.transAxes, fontsize=9, color=MUTED, va="bottom")
ax.set_axis_off()
fig.tight_layout()
fig.savefig(OUT / "fig1_map.png", bbox_inches="tight")

# ---------------- Figure 2: main result ----------------
r = pd.read_csv(OUT / "results.csv")
main = r[r.spec.str.startswith("A1")].set_index("var")
pct = lambda g: (1 - np.exp(g)) * 100
labels = {"diff_canton": "Cross a canton border\n(institutional)",
          "diff_lang": "Cross a language border\n(cultural)"}
fig, ax = plt.subplots(figsize=(8, 3.2), dpi=200)
for i, (v, lab) in enumerate(labels.items()):
    g, lo, hi = main.loc[v, ["coef", "lo", "hi"]]
    ax.barh(i, pct(g), color=["#5B6472", "#C8261B"][i], height=0.55)
    ax.errorbar(pct(g), i, xerr=[[pct(g) - pct(hi)], [pct(lo) - pct(g)]], color=INK, capsize=4, lw=1.2)
    ax.text(pct(lo) + 2, i, f"−{pct(g):.0f}%", va="center", fontsize=12, fontweight="bold", color=INK)
ax.set_yticks([0, 1], labels.values())
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xlabel("Reduction in commuting, holding travel time and commune size fixed (%)", color=MUTED)
ax.set_title("A language border cuts commuting far more than a canton border",
             loc="left", fontsize=12, fontweight="bold", color=INK)
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
ax.grid(axis="x", color="#e5e4e0", lw=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT / "fig2_result.png", bbox_inches="tight")
print("saved")

# ---------------- Summary table (used in the slides) ----------------
r = pd.read_csv(OUT / "results.csv")
out = []
for spec, g in r.groupby("spec", sort=False):
    g = g.set_index("var")
    d = g.coef.get("ln_time", np.nan)
    row = {"spec": spec, "n": int(g.n.iloc[0]), "time_elasticity": d}
    for v in ["diff_canton", "diff_lang", "de_to_fr", "fr_to_de", "involves_it", "involves_rm", "diff_canton:diff_lang"]:
        if v in g.index:
            row[f"{v}_coef"] = g.coef[v]
            row[f"{v}_se"] = g.se[v]
            row[f"{v}_pct_reduction"] = (1 - np.exp(g.coef[v])) * 100
            # the same effect expressed as extra travel time (multiplier on time)
            if not np.isnan(d):
                row[f"{v}_time_equiv_multiplier"] = np.exp(g.coef[v] / d)
    out.append(row)
summary = pd.DataFrame(out)
summary.to_csv(OUT / "summary.csv", index=False)
print(summary.round(3).to_string())
