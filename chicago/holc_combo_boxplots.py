"""
HOLC Redlining – Combined Box Plots for Chicago
=================================================
Plot 1: College Attainment & High School Completion vs HOLC Grade
Plot 2: Per Capita Income & Poverty Rate vs HOLC Grade (dual y-axes)
Plot 3: Life Expectancy vs HOLC Grade
"""

import matplotlib
matplotlib.use('Agg')
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import requests
import os
import warnings

warnings.filterwarnings("ignore")

# ─── Configuration ───────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))

COLOR_MAP = {
    "A": "#76a865",
    "B": "#7cbbe3",
    "C": "#ffff00",
    "D": "#d9534f",
}
GRADE_LABELS = {
    "A": '"Best"',
    "B": '"Still Desirable"',
    "C": '"Declining"',
    "D": '"Hazardous"',
}
GRADE_NUM = {"A": 1, "B": 2, "C": 3, "D": 4}

# Illinois FIPS = 17; Cook County = 031
STATE_FIPS = "17"
CHICAGO_COUNTIES = ["031"]

EDU_VARIABLES = (
    "NAME,"
    "B15003_001E,"
    "B15003_017E,B15003_018E,B15003_019E,B15003_020E,B15003_021E,"
    "B15003_022E,B15003_023E,B15003_024E,B15003_025E"
)


# ─── Helper functions ────────────────────────────────────────────────────────
def load_holc_data():
    cache_path = os.path.join(script_dir, "geojson.json")
    if not os.path.exists(cache_path):
        raise RuntimeError(f"HOLC GeoJSON not found at {cache_path}.")
    holc = gpd.read_file(cache_path)
    holc = holc[holc["grade"].isin(["A", "B", "C", "D"])].copy()
    return holc.to_crs(epsg=4326)


def load_tract_boundaries():
    cache_path = os.path.join(script_dir, "tracts_il.zip")
    if not os.path.exists(cache_path):
        print("  Downloading Illinois census tract boundaries...")
        urls = [
            f"https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_{STATE_FIPS}_tract_500k.zip",
            f"https://www2.census.gov/geo/tiger/GENZ2021/shp/cb_2021_{STATE_FIPS}_tract_500k.zip",
        ]
        for url in urls:
            try:
                resp = requests.get(url, timeout=120)
                if resp.status_code == 200:
                    with open(cache_path, "wb") as f:
                        f.write(resp.content)
                    break
            except Exception:
                pass
        else:
            raise RuntimeError("Failed to download census tract boundaries")
    tracts = gpd.read_file(cache_path)
    tracts = tracts[tracts["COUNTYFP"].isin(CHICAGO_COUNTIES)].copy()
    return tracts


def download_acs_education():
    all_rows, header = [], None
    for county in CHICAGO_COUNTIES:
        url = (
            f"https://api.census.gov/data/2022/acs/acs5"
            f"?get={EDU_VARIABLES}&for=tract:*"
            f"&in=state:{STATE_FIPS}&in=county:{county}"
        )
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                if header is None:
                    header = data[0]
                all_rows.extend(data[1:])
        except Exception:
            pass
    if not all_rows:
        raise RuntimeError("No ACS education data retrieved")
    df = pd.DataFrame(all_rows, columns=header)
    df["GEOID"] = df["state"] + df["county"] + df["tract"]
    for col in header:
        if col.startswith("B15003"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["total_25plus"] = df["B15003_001E"]
    hs_cols = ["B15003_017E","B15003_018E","B15003_019E","B15003_020E",
               "B15003_021E","B15003_022E","B15003_023E","B15003_024E","B15003_025E"]
    df["hs_or_more"] = df[hs_cols].sum(axis=1)
    bach_cols = ["B15003_022E","B15003_023E","B15003_024E","B15003_025E"]
    df["bachelors_plus"] = df[bach_cols].sum(axis=1)
    df["pct_hs_completed"] = np.where(df["total_25plus"] > 0,
        df["hs_or_more"] / df["total_25plus"] * 100, np.nan)
    df["pct_bachelors_plus"] = np.where(df["total_25plus"] > 0,
        df["bachelors_plus"] / df["total_25plus"] * 100, np.nan)
    df = df.dropna(subset=["pct_bachelors_plus", "pct_hs_completed"])
    df = df[df["total_25plus"] > 0]
    return df


def download_acs_income():
    variables = "NAME,B19301_001E,B17001_001E,B17001_002E,B19013_001E"
    all_rows, header = [], None
    for county in CHICAGO_COUNTIES:
        url = (
            f"https://api.census.gov/data/2022/acs/acs5"
            f"?get={variables}&for=tract:*"
            f"&in=state:{STATE_FIPS}&in=county:{county}"
        )
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                if header is None:
                    header = data[0]
                all_rows.extend(data[1:])
        except Exception:
            pass
    if not all_rows:
        raise RuntimeError("No ACS income data retrieved")
    df = pd.DataFrame(all_rows, columns=header)
    df["GEOID"] = df["state"] + df["county"] + df["tract"]
    df["per_capita_income"] = pd.to_numeric(df["B19301_001E"], errors="coerce")
    df["poverty_total"] = pd.to_numeric(df["B17001_001E"], errors="coerce")
    df["poverty_below"] = pd.to_numeric(df["B17001_002E"], errors="coerce")
    df["poverty_rate"] = np.where(df["poverty_total"] > 0,
        df["poverty_below"] / df["poverty_total"] * 100, np.nan)
    df = df.dropna(subset=["per_capita_income"])
    df = df[df["per_capita_income"] > 0]
    return df


def compute_dominant_grade_tracts(holc, tracts):
    holc_proj = holc.to_crs(epsg=3857)
    tracts_proj = tracts.to_crs(epsg=3857)
    overlay = gpd.overlay(tracts_proj, holc_proj, how="intersection")
    overlay["overlap_area"] = overlay.geometry.area
    results = []
    for geoid in tracts["GEOID"].unique():
        area_overlaps = overlay[overlay["GEOID"] == geoid]
        if len(area_overlaps) == 0:
            continue
        grade_areas = area_overlaps.groupby("grade")["overlap_area"].sum()
        total_overlap = grade_areas.sum()
        if total_overlap == 0:
            continue
        results.append({
            "GEOID": geoid,
            "dominant_grade": grade_areas.idxmax(),
        })
    return pd.DataFrame(results)


def compute_dominant_grade_comm(holc, comm_areas):
    holc_proj = holc.to_crs(epsg=3857)
    comm_proj = comm_areas.to_crs(epsg=3857)
    overlay = gpd.overlay(comm_proj, holc_proj, how="intersection")
    overlay["overlap_area"] = overlay.geometry.area
    results = []
    for area_num in comm_areas["area_numbe"].unique():
        area_overlaps = overlay[overlay["area_numbe"] == area_num]
        if len(area_overlaps) == 0:
            continue
        grade_areas = area_overlaps.groupby("grade")["overlap_area"].sum()
        total_overlap = grade_areas.sum()
        if total_overlap == 0:
            continue
        results.append({
            "area_numbe": area_num,
            "dominant_grade": grade_areas.idxmax(),
        })
    return pd.DataFrame(results)


def style_boxplot(bp, color, alpha=0.65):
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(alpha)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.2)
    for whisker in bp["whiskers"]:
        whisker.set_color("black")
        whisker.set_linewidth(1.1)
    for cap in bp["caps"]:
        cap.set_color("black")
        cap.set_linewidth(1.1)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(2)
    for flier in bp["fliers"]:
        flier.set(marker="o", markerfacecolor="gray", alpha=0.4, markersize=4)


# ─── Load education & income data (census tracts) ───────────────────────────
print("Loading data...")
holc = load_holc_data()

print("  Loading census tracts...")
tracts = load_tract_boundaries()
tracts = tracts.to_crs(epsg=4326)
tracts["GEOID"] = tracts["GEOID"].astype(str)

print("  Downloading ACS education data...")
edu_data = download_acs_education()

print("  Downloading ACS income data...")
income_data = download_acs_income()

tracts_edu = tracts.merge(
    edu_data[["GEOID", "pct_bachelors_plus", "pct_hs_completed"]], on="GEOID", how="left"
).dropna(subset=["pct_bachelors_plus"])

tracts_inc = tracts.merge(
    income_data[["GEOID", "per_capita_income", "poverty_rate"]], on="GEOID", how="left"
).dropna(subset=["per_capita_income"])

print("  Computing HOLC grade overlays (census tracts)...")
grade_edu = compute_dominant_grade_tracts(holc, tracts_edu)
grade_inc = compute_dominant_grade_tracts(holc, tracts_inc)

edu_analysis = tracts_edu.merge(grade_edu, on="GEOID", how="inner").dropna(subset=["pct_bachelors_plus"])
inc_analysis = tracts_inc.merge(grade_inc, on="GEOID", how="inner").dropna(subset=["per_capita_income"])

print(f"  Education analysis: {len(edu_analysis)} tracts")
print(f"  Income analysis:    {len(inc_analysis)} tracts")


# ─── Load life expectancy data (community areas) ────────────────────────────
print("\n  Downloading community area boundaries...")
comm_url = "https://data.cityofchicago.org/resource/igwz-8jzy.geojson?$limit=100"
comm_areas = gpd.read_file(comm_url)
comm_areas = comm_areas.to_crs(epsg=4326)
comm_areas["area_numbe"] = comm_areas["area_numbe"].astype(int)

print("  Downloading life expectancy data...")
le_url = "https://data.cityofchicago.org/resource/qjr3-bm53.json?$limit=100"
le_resp = requests.get(le_url, timeout=30)
le_data = pd.DataFrame(le_resp.json())
le_data["ca"] = pd.to_numeric(le_data["ca"], errors="coerce")
le_data["life_expectancy"] = pd.to_numeric(le_data["_2010_life_expectancy"], errors="coerce")
le_data = le_data[["ca", "life_expectancy"]].dropna()
le_data["ca"] = le_data["ca"].astype(int)

comm_areas = comm_areas.merge(le_data, left_on="area_numbe", right_on="ca", how="left")
comm_areas = comm_areas.dropna(subset=["life_expectancy"])

print("  Computing HOLC grade overlays (community areas)...")
grade_le = compute_dominant_grade_comm(holc, comm_areas)
le_analysis = comm_areas.merge(grade_le, on="area_numbe", how="inner").dropna(subset=["life_expectancy"])
print(f"  Life expectancy analysis: {len(le_analysis)} community areas")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Education: College Attainment & HS Completion by HOLC Grade
# ═══════════════════════════════════════════════════════════════════════════════
print("\nGenerating education combo box plot...")

grades = ["A", "B", "C", "D"]
fig, ax = plt.subplots(figsize=(12, 7))

positions_college, positions_hs = [], []
data_college, data_hs = [], []
tick_positions, tick_labels = [], []
width = 0.35
gap = 1.0

for i, g in enumerate(grades):
    subset = edu_analysis[edu_analysis["dominant_grade"] == g]
    if len(subset) == 0:
        continue
    center = i * (2 * width + gap)
    positions_college.append(center - width / 2 - 0.02)
    positions_hs.append(center + width / 2 + 0.02)
    data_college.append(subset["pct_bachelors_plus"].values)
    data_hs.append(subset["pct_hs_completed"].values)
    tick_positions.append(center)
    tick_labels.append(f"Grade {g}\n{GRADE_LABELS[g]}\n(n={len(subset)})")

bp1 = ax.boxplot(data_college, positions=positions_college, widths=width,
                 patch_artist=True, zorder=3)
style_boxplot(bp1, "#4C72B0")

bp2 = ax.boxplot(data_hs, positions=positions_hs, widths=width,
                 patch_artist=True, zorder=3)
style_boxplot(bp2, "#DD8452")

ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, fontsize=10, fontweight="medium")
ax.set_ylabel("Percentage of Adults (%)", fontsize=12, fontweight="bold")
ax.set_title("College Attainment & High School Completion\nby HOLC Redlining Grade — Chicago",
             fontsize=15, fontweight="bold", pad=15)

ax.legend(
    [bp1["boxes"][0], bp2["boxes"][0]],
    ["Bachelor's Degree or Higher", "Completed High School or Higher"],
    loc="upper right", fontsize=10, framealpha=0.9, edgecolor="gray",
)

ax.yaxis.grid(True, linestyle="--", alpha=0.4, color="gray")
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.2)
ax.spines["bottom"].set_linewidth(1.2)

plt.tight_layout()
out_edu = os.path.join(script_dir, "holc_education_combo_boxplot.png")
plt.savefig(out_edu, dpi=300, bbox_inches="tight")
print(f"  Saved {out_edu}")
plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Income & Poverty Rate by HOLC Grade (dual y-axes)
# ═══════════════════════════════════════════════════════════════════════════════
print("\nGenerating income/poverty combo box plot...")

fig2, ax_inc = plt.subplots(figsize=(12, 7))
ax_pov = ax_inc.twinx()

positions_income, positions_poverty = [], []
data_income, data_poverty = [], []
tick_positions2, tick_labels2 = [], []

for i, g in enumerate(grades):
    subset = inc_analysis[inc_analysis["dominant_grade"] == g]
    if len(subset) == 0:
        continue
    center = i * (2 * width + gap)
    positions_income.append(center - width / 2 - 0.02)
    positions_poverty.append(center + width / 2 + 0.02)
    data_income.append(subset["per_capita_income"].values / 1000)
    pov_vals = subset["poverty_rate"].dropna().values
    data_poverty.append(pov_vals if len(pov_vals) > 0 else [np.nan])
    tick_positions2.append(center)
    tick_labels2.append(f"Grade {g}\n{GRADE_LABELS[g]}\n(n={len(subset)})")

bp_inc = ax_inc.boxplot(data_income, positions=positions_income, widths=width,
                        patch_artist=True, zorder=3)
style_boxplot(bp_inc, "#2a9d8f")

bp_pov = ax_pov.boxplot(data_poverty, positions=positions_poverty, widths=width,
                        patch_artist=True, zorder=3)
style_boxplot(bp_pov, "#e76f51")

ax_inc.set_xticks(tick_positions2)
ax_inc.set_xticklabels(tick_labels2, fontsize=10, fontweight="medium")

ax_inc.set_ylabel("Per Capita Income ($K)", fontsize=12, fontweight="bold", color="#2a9d8f")
ax_pov.set_ylabel("Poverty Rate (%)", fontsize=12, fontweight="bold", color="#e76f51")

ax_inc.tick_params(axis="y", labelcolor="#2a9d8f")
ax_pov.tick_params(axis="y", labelcolor="#e76f51")

ax_inc.set_title("Per Capita Income & Poverty Rate\nby HOLC Redlining Grade — Chicago",
                 fontsize=15, fontweight="bold", pad=15)

ax_inc.legend(
    [bp_inc["boxes"][0], bp_pov["boxes"][0]],
    ["Per Capita Income ($K)", "Poverty Rate (%)"],
    loc="upper right", fontsize=10, framealpha=0.9, edgecolor="gray",
)

ax_inc.yaxis.grid(True, linestyle="--", alpha=0.4, color="gray")
ax_inc.set_axisbelow(True)
ax_inc.spines["top"].set_visible(False)
ax_inc.spines["left"].set_linewidth(1.2)
ax_inc.spines["bottom"].set_linewidth(1.2)
ax_pov.spines["top"].set_visible(False)

plt.tight_layout()
out_inc = os.path.join(script_dir, "holc_income_poverty_combo_boxplot.png")
plt.savefig(out_inc, dpi=300, bbox_inches="tight")
print(f"  Saved {out_inc}")
plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Life Expectancy by HOLC Grade
# ═══════════════════════════════════════════════════════════════════════════════
print("\nGenerating life expectancy box plot...")

fig3, ax3 = plt.subplots(figsize=(10, 7))

grade_data, grade_labels_le, grade_colors = [], [], []
for g in grades:
    subset = le_analysis[le_analysis["dominant_grade"] == g]["life_expectancy"]
    if len(subset) > 0:
        grade_data.append(subset.values)
        grade_labels_le.append(f"Grade {g}\n{GRADE_LABELS[g]}\n(n={len(subset)})")
        grade_colors.append(COLOR_MAP[g])

bp3 = ax3.boxplot(grade_data, labels=grade_labels_le, patch_artist=True, widths=0.55)
for patch, color in zip(bp3["boxes"], grade_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
    patch.set_edgecolor("black")
    patch.set_linewidth(1.2)
for whisker in bp3["whiskers"]:
    whisker.set_color("black")
    whisker.set_linewidth(1.1)
for cap in bp3["caps"]:
    cap.set_color("black")
    cap.set_linewidth(1.1)
for median in bp3["medians"]:
    median.set_color("black")
    median.set_linewidth(2)
for flier in bp3["fliers"]:
    flier.set(marker="o", markerfacecolor="gray", alpha=0.4, markersize=4)

ax3.set_ylabel("Life Expectancy (Years)", fontsize=12, fontweight="bold")
ax3.set_title("Life Expectancy by HOLC Redlining Grade — Chicago",
              fontsize=15, fontweight="bold", pad=15)

ax3.yaxis.grid(True, linestyle="--", alpha=0.4, color="gray")
ax3.set_axisbelow(True)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.spines["left"].set_linewidth(1.2)
ax3.spines["bottom"].set_linewidth(1.2)

plt.tight_layout()
out_le = os.path.join(script_dir, "holc_life_expectancy_boxplot.png")
plt.savefig(out_le, dpi=300, bbox_inches="tight")
print(f"  Saved {out_le}")
plt.show()

print("\nDone! All three box plots saved.")
