"""
HOLC Redlining and Life Expectancy Correlation Analysis – Atlanta, Georgia
=========================================================================
This script analyzes the correlation between 1930s HOLC redlining grades
in Atlanta and current life expectancy by census tract.

Data sources:
- HOLC redlining zones: Mapping Inequality (University of Richmond)
- Life expectancy: CDC/NCHS USALEEP (U.S. Small-Area Life Expectancy
  Estimates Project, 2010-2015) — census-tract-level e(0) estimates
- Census tract boundaries: Census Bureau TIGER/Line Cartographic Boundaries
"""

import matplotlib
matplotlib.use('Agg')
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats
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

GRADE_NUM = {"A": 1, "B": 2, "C": 3, "D": 4}

STATE_FIPS = "13"
ATLANTA_COUNTIES = ["121", "089"]


# ─── Helper functions ────────────────────────────────────────────────────────
def download_holc_data():
    """Download and cache Atlanta HOLC redlining GeoJSON."""
    cache_path = os.path.join(script_dir, "geojson.json")
    if not os.path.exists(cache_path):
        print("  Downloading Atlanta HOLC data from Mapping Inequality...")
        urls = [
            "https://dsl.richmond.edu/panorama/redlining/static/citiesData/GAAtlanta1938/geojson.json",
            "https://dsl.richmond.edu/panorama/redlining/static/downloads/geojson/GAAtlanta1938.geojson",
        ]
        headers = {"Accept": "application/json", "User-Agent": "Mozilla/5.0"}
        for url in urls:
            try:
                resp = requests.get(url, timeout=60, headers=headers)
                if resp.status_code == 200 and len(resp.content) > 1000 and resp.text.lstrip().startswith('{'):
                    with open(cache_path, "wb") as f:
                        f.write(resp.content)
                    print(f"    Saved to {cache_path}")
                    break
            except Exception as e:
                print(f"    Warning: {url} → {e}")
        else:
            raise RuntimeError(
                "Could not download Atlanta HOLC data.\n"
                "Download from https://dsl.richmond.edu/panorama/redlining/"
            )
    return gpd.read_file(cache_path)


def download_tract_boundaries():
    """Download census tract boundaries for Georgia, filter to Atlanta."""
    cache_path = os.path.join(script_dir, "tracts_ga.zip")
    urls = [
        f"https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_{STATE_FIPS}_tract_500k.zip",
        f"https://www2.census.gov/geo/tiger/GENZ2021/shp/cb_2021_{STATE_FIPS}_tract_500k.zip",
    ]
    if not os.path.exists(cache_path):
        print("  Downloading Georgia census tract boundaries...")
        for url in urls:
            try:
                resp = requests.get(url, timeout=120)
                if resp.status_code == 200:
                    with open(cache_path, "wb") as f:
                        f.write(resp.content)
                    print(f"    Saved to {cache_path}")
                    break
            except Exception as e:
                print(f"    Warning: {url} → {e}")
        else:
            raise RuntimeError("Failed to download census tract boundaries")

    tracts = gpd.read_file(cache_path)
    tracts = tracts[tracts["COUNTYFP"].isin(ATLANTA_COUNTIES)].copy()
    return tracts


def download_life_expectancy():
    """
    Download life expectancy data from CDC USALEEP (Socrata).

    The USALEEP dataset provides census-tract-level life expectancy at birth
    (e(0)) based on 2010-2015 mortality data on 2010 census geography.
    Resource ID on data.cdc.gov: 5h56-n989

    Columns: state_name, county_name, full_ct_num, le, le_range, se_le
    """
    print("  Querying CDC USALEEP data for Georgia...")

    # County names as they appear in the USALEEP dataset
    COUNTY_NAMES = {
        "121": "Fulton County, GA",
        "089": "DeKalb County, GA",
    }

    all_records = []
    for county_fips, county_name in COUNTY_NAMES.items():
        url = (
            "https://data.cdc.gov/resource/5h56-n989.json"
            f"?$limit=5000"
            f"&state_name=Georgia"
            f"&county_name={requests.utils.quote(county_name)}"
        )
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                records = resp.json()
                # Tag each record with the county FIPS for GEOID construction
                for rec in records:
                    rec["_county_fips"] = county_fips
                all_records.extend(records)
                print(f"    {county_name}: {len(records)} tracts")
        except Exception as e:
            print(f"    Warning: {county_name} → {e}")

    if not all_records:
        raise RuntimeError(
            "Could not download USALEEP life expectancy data from data.cdc.gov"
        )

    df = pd.DataFrame(all_records)

    # Life expectancy column is "le"
    df["life_expectancy"] = pd.to_numeric(df["le"], errors="coerce")

    # Build 11-digit GEOID from state FIPS + county FIPS + full_ct_num
    # full_ct_num is like "0001.00" — remove the dot to get 6-digit tract code
    df["tract_code"] = (
        df["full_ct_num"]
        .astype(str)
        .str.replace(".", "", regex=False)
        .str.zfill(6)
    )
    df["GEOID_2010"] = STATE_FIPS + df["_county_fips"] + df["tract_code"]

    df = df.dropna(subset=["life_expectancy"])
    df = df[df["life_expectancy"] > 0]

    return df


# ─── 1. Load HOLC redlining data ────────────────────────────────────────────
print("Loading HOLC redlining data...")
holc = download_holc_data()
holc = holc[holc["grade"].isin(["A", "B", "C", "D"])].copy()
holc = holc.to_crs(epsg=4326)

print(f"  Loaded {len(holc)} HOLC zones")
for g in ["A", "B", "C", "D"]:
    print(f"    Grade {g}: {(holc['grade'] == g).sum()} zones")


# ─── 2. Download census tract boundaries ────────────────────────────────────
print("\nDownloading Atlanta-area census tract boundaries...")
tracts = download_tract_boundaries()
tracts = tracts.to_crs(epsg=4326)
tracts["GEOID"] = tracts["GEOID"].astype(str)
print(f"  {len(tracts)} census tracts")


# ─── 3. Download life expectancy data ───────────────────────────────────────
print("\nDownloading life expectancy data (CDC USALEEP)...")
le_data = download_life_expectancy()

# USALEEP uses 2010 census tracts; our boundaries are 2020.
# Most tract GEOIDs persist — do a best-effort join.
# Try joining on the first 11 characters (full GEOID)
print(f"  USALEEP: {len(le_data)} tracts with life expectancy data")
print(
    f"  Range: {le_data['life_expectancy'].min():.1f} – "
    f"{le_data['life_expectancy'].max():.1f} years"
)


# ─── 4. Merge tracts with life expectancy ───────────────────────────────────
print("\nMerging census tracts with life expectancy data...")

# Join 2010 USALEEP GEOIDs to 2020 tract GEOIDs (best-effort match)
tracts = tracts.merge(
    le_data[["GEOID_2010", "life_expectancy"]],
    left_on="GEOID",
    right_on="GEOID_2010",
    how="left",
)
matched = tracts["life_expectancy"].notna().sum()
print(f"  Matched {matched} of {len(tracts)} tracts (2010→2020 GEOID join)")

tracts = tracts.dropna(subset=["life_expectancy"])

if len(tracts) < 5:
    print("WARNING: Very few tracts matched. This may be due to 2010→2020 tract ID changes.")
    print("  Attempting fuzzy match on 9-digit prefix...")
    # Fallback: match on state+county+first 4 digits of tract
    le_data["GEOID_prefix"] = le_data["GEOID_2010"].str[:9]
    tracts_orig = download_tract_boundaries()
    tracts_orig = tracts_orig.to_crs(epsg=4326)
    tracts_orig["GEOID"] = tracts_orig["GEOID"].astype(str)
    tracts_orig["GEOID_prefix"] = tracts_orig["GEOID"].str[:9]
    # Average LE by prefix
    le_prefix = le_data.groupby("GEOID_prefix")["life_expectancy"].mean().reset_index()
    tracts = tracts_orig.merge(le_prefix, on="GEOID_prefix", how="inner")
    print(f"  Prefix match: {len(tracts)} tracts")

print(f"  Final: {len(tracts)} tracts with life expectancy data")


# ─── 5. Spatial overlay: find dominant HOLC grade per census tract ───────────
print("\nCalculating dominant HOLC grade per census tract...")

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

    dominant_grade = grade_areas.idxmax()
    weighted_score = sum(
        GRADE_NUM[g] * (a / total_overlap) for g, a in grade_areas.items()
    )
    grade_fracs = {
        g: grade_areas.get(g, 0) / total_overlap for g in ["A", "B", "C", "D"]
    }
    redlined_frac = grade_fracs["D"]

    results.append(
        {
            "GEOID": geoid,
            "dominant_grade": dominant_grade,
            "weighted_grade_score": weighted_score,
            "redlined_fraction": redlined_frac,
            "grade_A_frac": grade_fracs["A"],
            "grade_B_frac": grade_fracs["B"],
            "grade_C_frac": grade_fracs["C"],
            "grade_D_frac": grade_fracs["D"],
            "total_holc_overlap": total_overlap,
        }
    )

results_df = pd.DataFrame(results)
print(f"  Matched {len(results_df)} census tracts with HOLC data")

# Merge with life expectancy
analysis = tracts.merge(results_df, on="GEOID", how="inner")
analysis = analysis.dropna(subset=["life_expectancy", "weighted_grade_score"])
print(f"  Final analysis dataset: {len(analysis)} census tracts")


# ─── 6. Statistical analysis ────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  STATISTICAL RESULTS — HOLC Grade vs Life Expectancy (Atlanta)")
print("=" * 65)

# Correlation: weighted grade score vs life expectancy
r_weighted, p_weighted = stats.pearsonr(
    analysis["weighted_grade_score"], analysis["life_expectancy"]
)
print(f"\n  Pearson correlation (weighted HOLC score vs life expectancy):")
print(f"    r = {r_weighted:.4f},  p = {p_weighted:.6f}")
if p_weighted < 0.05:
    print(f"    → Statistically significant (p < 0.05)")

# Correlation: redlined fraction vs life expectancy
r_redlined, p_redlined = stats.pearsonr(
    analysis["redlined_fraction"], analysis["life_expectancy"]
)
print(f"\n  Pearson correlation (redlined fraction vs life expectancy):")
print(f"    r = {r_redlined:.4f},  p = {p_redlined:.6f}")
if p_redlined < 0.05:
    print(f"    → Statistically significant (p < 0.05)")

# Mean life expectancy by dominant HOLC grade
print(f"\n  Mean life expectancy by dominant HOLC grade:")
print(f"  {'Grade':<8} {'Mean LE':<10} {'Median LE':<10} {'Count':<6}")
print(f"  {'-' * 34}")
for grade in ["A", "B", "C", "D"]:
    subset = analysis[analysis["dominant_grade"] == grade]
    if len(subset) > 0:
        mean_le = subset["life_expectancy"].mean()
        median_le = subset["life_expectancy"].median()
        print(
            f"  {grade:<8} {mean_le:<10.1f} {median_le:<10.1f} {len(subset):<6}"
        )

# ANOVA
groups = [
    analysis[analysis["dominant_grade"] == g]["life_expectancy"].values
    for g in ["A", "B", "C", "D"]
    if len(analysis[analysis["dominant_grade"] == g]) > 0
]
if len(groups) >= 2:
    f_stat, p_anova = stats.f_oneway(*groups)
    print(f"\n  One-way ANOVA (life expectancy across HOLC grades):")
    print(f"    F = {f_stat:.4f},  p = {p_anova:.6f}")
    if p_anova < 0.05:
        print(f"    → Significant difference between groups (p < 0.05)")

# Cohen's d (Grade A vs Grade D)
a_le = analysis[analysis["dominant_grade"] == "A"]["life_expectancy"]
d_le = analysis[analysis["dominant_grade"] == "D"]["life_expectancy"]
if len(a_le) > 1 and len(d_le) > 1:
    pooled_std = np.sqrt(
        ((len(a_le) - 1) * a_le.std() ** 2 + (len(d_le) - 1) * d_le.std() ** 2)
        / (len(a_le) + len(d_le) - 2)
    )
    if pooled_std > 0:
        cohens_d = (a_le.mean() - d_le.mean()) / pooled_std
        print(f"\n  Cohen's d (Grade A vs Grade D):")
        print(f"    d = {cohens_d:.4f}")
        if abs(cohens_d) >= 0.8:
            print(f"    → Large effect size")
        elif abs(cohens_d) >= 0.5:
            print(f"    → Medium effect size")
        elif abs(cohens_d) >= 0.2:
            print(f"    → Small effect size")

print("=" * 65)


# ─── 6b. Export statistics to CSV ────────────────────────────────────────────
stats_rows = []

stats_rows.append({
    "Analysis": "Life Expectancy (Atlanta)",
    "Test": "Pearson (weighted HOLC score vs life expectancy)",
    "Statistic": f"r = {r_weighted:.4f}",
    "p-value": f"{p_weighted:.6f}",
})
stats_rows.append({
    "Analysis": "Life Expectancy (Atlanta)",
    "Test": "Pearson (redlined fraction vs life expectancy)",
    "Statistic": f"r = {r_redlined:.4f}",
    "p-value": f"{p_redlined:.6f}",
})
if len(groups) >= 2:
    stats_rows.append({
        "Analysis": "Life Expectancy (Atlanta)",
        "Test": "One-way ANOVA (life expectancy across HOLC grades)",
        "Statistic": f"F = {f_stat:.4f}",
        "p-value": f"{p_anova:.6f}",
    })
if len(a_le) > 1 and len(d_le) > 1 and pooled_std > 0:
    stats_rows.append({
        "Analysis": "Life Expectancy (Atlanta)",
        "Test": "Cohen's d (Grade A vs Grade D)",
        "Statistic": f"d = {cohens_d:.4f}",
        "p-value": "N/A",
    })

stats_df = pd.DataFrame(stats_rows)
stats_csv_path = os.path.join(script_dir, "holc_life_expectancy_statistics.csv")
stats_df.to_csv(stats_csv_path, index=False)
print(f"\n  Statistics saved to {stats_csv_path}")


# ─── 7. Visualization (separate images) ─────────────────────────────────────
print("\nGenerating visualizations...")

# --- Plot 1: Scatter – weighted grade score vs life expectancy ---
fig1, ax1 = plt.subplots(figsize=(10, 8))
colors = [COLOR_MAP[g] for g in analysis["dominant_grade"]]
ax1.scatter(
    analysis["weighted_grade_score"],
    analysis["life_expectancy"],
    c=colors,
    edgecolors="black",
    s=80,
    alpha=0.8,
    zorder=5,
)

slope, intercept, r, p, se = stats.linregress(
    analysis["weighted_grade_score"], analysis["life_expectancy"]
)
x_line = np.linspace(1, 4, 100)
ax1.plot(x_line, slope * x_line + intercept, "k--", linewidth=2, alpha=0.7)
ax1.set_xlabel(
    "Area-Weighted HOLC Grade Score\n(1=A/Best → 4=D/Hazardous)", fontsize=11
)
ax1.set_ylabel("Life Expectancy (years)", fontsize=11)
ax1.set_title(
    f"HOLC Score vs Life Expectancy in Atlanta\nr={r:.3f}, p={p:.4f}",
    fontsize=13,
    fontweight="bold",
)
ax1.set_xticks([1, 2, 3, 4])
ax1.set_xticklabels(["1 (A)", "2 (B)", "3 (C)", "4 (D)"])
ax1.grid(True, alpha=0.3)
plt.tight_layout()
out1 = os.path.join(script_dir, "holc_life_scatter.png")
plt.savefig(out1, dpi=300, bbox_inches="tight")
print(f"  Saved {out1}")
plt.show()

# --- Plot 2: Box plot – life expectancy by dominant grade ---
fig2, ax2 = plt.subplots(figsize=(10, 5))
grade_data = []
grade_labels = []
grade_colors = []
for g in ["A", "B", "C", "D"]:
    subset = analysis[analysis["dominant_grade"] == g]["life_expectancy"]
    if len(subset) > 0:
        grade_data.append(subset.values)
        grade_labels.append(f"Grade {g}")
        grade_colors.append(COLOR_MAP[g])

bp = ax2.boxplot(grade_data, labels=grade_labels, patch_artist=True, widths=0.6)
for patch, color in zip(bp["boxes"], grade_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax2.set_ylabel("Life Expectancy (years)", fontsize=11)
ax2.set_title(
    "Life Expectancy by Dominant HOLC Grade – Atlanta",
    fontsize=13,
    fontweight="bold",
)
ax2.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
out2 = os.path.join(script_dir, "holc_life_boxplot.png")
plt.savefig(out2, dpi=300, bbox_inches="tight")
print(f"  Saved {out2}")
plt.show()

# --- Plot 3: Scatter – redlined fraction vs life expectancy ---
fig3, ax3 = plt.subplots(figsize=(10, 8))
ax3.scatter(
    analysis["redlined_fraction"] * 100,
    analysis["life_expectancy"],
    c=colors,
    edgecolors="black",
    s=80,
    alpha=0.8,
    zorder=5,
)
slope2, intercept2, r2, p2, se2 = stats.linregress(
    analysis["redlined_fraction"] * 100, analysis["life_expectancy"]
)
x_line2 = np.linspace(0, 100, 100)
ax3.plot(x_line2, slope2 * x_line2 + intercept2, "k--", linewidth=2, alpha=0.7)
ax3.set_xlabel("% of Census Tract Redlined (Grade D)", fontsize=11)
ax3.set_ylabel("Life Expectancy (years)", fontsize=11)
ax3.set_title(
    f"Redlined Fraction vs Life Expectancy in Atlanta\nr={r2:.3f}, p={p2:.4f}",
    fontsize=13,
    fontweight="bold",
)
ax3.grid(True, alpha=0.3)
plt.tight_layout()
out3 = os.path.join(script_dir, "holc_life_redlined_scatter.png")
plt.savefig(out3, dpi=300, bbox_inches="tight")
print(f"  Saved {out3}")
plt.show()

# --- Plot 4: Choropleth map – life expectancy with HOLC overlay ---
fig4, ax4 = plt.subplots(figsize=(12, 10))
analysis_geo = analysis.copy()
analysis_geo = analysis_geo.to_crs(epsg=3857)
analysis_geo.plot(
    column="life_expectancy",
    cmap="RdYlGn",
    legend=True,
    ax=ax4,
    edgecolor="gray",
    linewidth=0.5,
    alpha=0.8,
    legend_kwds={"label": "Life Expectancy (years)", "shrink": 0.7},
)

holc_plot = holc.to_crs(epsg=3857)
for grade in ["D", "C", "B", "A"]:
    subset = holc_plot[holc_plot["grade"] == grade]
    subset.boundary.plot(ax=ax4, color=COLOR_MAP[grade], linewidth=0.4, alpha=0.5)

ax4.set_axis_off()
ax4.set_title(
    "Life Expectancy Map with HOLC Zone Boundaries – Atlanta",
    fontsize=13,
    fontweight="bold",
)

legend_elements = [
    Patch(facecolor=c, edgecolor="black", label=f"HOLC Grade {g}")
    for g, c in COLOR_MAP.items()
]
ax4.legend(handles=legend_elements, loc="lower left", fontsize=9)
plt.tight_layout()
out4 = os.path.join(script_dir, "holc_life_choropleth.png")
plt.savefig(out4, dpi=300, bbox_inches="tight")
print(f"  Saved {out4}")
plt.show()

print("\nDone!")
