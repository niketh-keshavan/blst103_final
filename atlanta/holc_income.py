"""
HOLC Redlining and Income/Wages Correlation Analysis – Atlanta, Georgia
=======================================================================
This script analyzes the correlation between 1930s HOLC redlining grades
in Atlanta and current income levels by census tract.

Data sources:
- HOLC redlining zones: Mapping Inequality (University of Richmond)
- Per capita income & poverty: U.S. Census Bureau ACS 5-Year Estimates
  (Tables B19301, B17001, B19013)
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

# HOLC grade color mapping
COLOR_MAP = {
    "A": "#76a865",  # Green  – "Best"
    "B": "#7cbbe3",  # Blue   – "Still Desirable"
    "C": "#ffff00",  # Yellow – "Definitely Declining"
    "D": "#d9534f",  # Red    – "Hazardous" (redlined)
}

# Numeric encoding for correlation (higher = worse grade)
GRADE_NUM = {"A": 1, "B": 2, "C": 3, "D": 4}

# Georgia FIPS = 13; Atlanta-area counties: Fulton (121), DeKalb (089)
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
                "Download manually from https://dsl.richmond.edu/panorama/redlining/\n"
                f"and save as {cache_path}"
            )
    return gpd.read_file(cache_path)


def download_tract_boundaries():
    """Download census tract boundaries for Georgia, filter to Atlanta area."""
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
    # Filter to Atlanta-area counties
    tracts = tracts[tracts["COUNTYFP"].isin(ATLANTA_COUNTIES)].copy()
    return tracts


def download_acs_income():
    """Download per capita income and poverty data from Census ACS API."""
    variables = "NAME,B19301_001E,B17001_001E,B17001_002E,B19013_001E"
    all_rows = []
    header = None

    for county in ATLANTA_COUNTIES:
        url = (
            f"https://api.census.gov/data/2022/acs/acs5"
            f"?get={variables}"
            f"&for=tract:*"
            f"&in=state:{STATE_FIPS}&in=county:{county}"
        )
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                if header is None:
                    header = data[0]
                all_rows.extend(data[1:])
                print(f"    County {county}: {len(data) - 1} tracts")
            else:
                print(f"    Warning: county {county} → HTTP {resp.status_code}")
        except Exception as e:
            print(f"    Warning: county {county} → {e}")

    if not all_rows:
        raise RuntimeError("No ACS income data retrieved")

    df = pd.DataFrame(all_rows, columns=header)

    # Construct 11-digit GEOID (state + county + tract)
    df["GEOID"] = df["state"] + df["county"] + df["tract"]

    # Parse numeric columns
    df["per_capita_income"] = pd.to_numeric(df["B19301_001E"], errors="coerce")
    df["poverty_total"] = pd.to_numeric(df["B17001_001E"], errors="coerce")
    df["poverty_below"] = pd.to_numeric(df["B17001_002E"], errors="coerce")
    df["median_household_income"] = pd.to_numeric(df["B19013_001E"], errors="coerce")

    # Compute poverty rate
    df["poverty_rate"] = np.where(
        df["poverty_total"] > 0,
        df["poverty_below"] / df["poverty_total"] * 100,
        np.nan,
    )

    df = df.dropna(subset=["per_capita_income"])
    df = df[df["per_capita_income"] > 0]
    return df


# ─── 1. Load HOLC redlining data ────────────────────────────────────────────
print("Loading HOLC redlining data...")
holc = download_holc_data()

# Keep only residential zones with a valid grade (A-D)
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
print(f"  {len(tracts)} census tracts in Fulton + DeKalb counties")


# ─── 3. Download income / socioeconomic data ────────────────────────────────
print("\nDownloading income and poverty data from Census ACS 5-Year...")
income_data = download_acs_income()
print(f"  Income data for {len(income_data)} census tracts")
print(
    f"  Per capita income range: "
    f"${income_data['per_capita_income'].min():,.0f} – "
    f"${income_data['per_capita_income'].max():,.0f}"
)
if income_data["poverty_rate"].notna().sum() > 0:
    print(
        f"  Poverty rate range: "
        f"{income_data['poverty_rate'].min():.1f}% – "
        f"{income_data['poverty_rate'].max():.1f}%"
    )


# ─── 4. Merge tracts with income data ───────────────────────────────────────
print("\nMerging census tracts with income data...")
tracts = tracts.merge(
    income_data[["GEOID", "per_capita_income", "poverty_rate", "median_household_income"]],
    on="GEOID",
    how="left",
)
tracts = tracts.dropna(subset=["per_capita_income"])
print(f"  {len(tracts)} tracts with valid income data")


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

# Merge with income data
analysis = tracts.merge(results_df, on="GEOID", how="inner")
analysis = analysis.dropna(subset=["per_capita_income", "weighted_grade_score"])
print(f"  Final analysis dataset: {len(analysis)} census tracts")


# ─── 6. Statistical analysis ────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  STATISTICAL RESULTS — HOLC Grade vs Income / Wages (Atlanta)")
print("=" * 70)

# --- Pearson correlation: weighted grade score vs per capita income ---
r_weighted, p_weighted = stats.pearsonr(
    analysis["weighted_grade_score"], analysis["per_capita_income"]
)
print(f"\n  Pearson correlation (weighted HOLC score vs per capita income):")
print(f"    r = {r_weighted:.4f},  p = {p_weighted:.6f}")
if p_weighted < 0.05:
    print(f"    → Statistically significant (p < 0.05)")
if r_weighted < 0:
    print(f"    → Negative: worse HOLC grades associated with LOWER income")
else:
    print(f"    → Positive: worse HOLC grades associated with higher income")

# --- Spearman rank correlation ---
r_spear, p_spear = stats.spearmanr(
    analysis["weighted_grade_score"], analysis["per_capita_income"]
)
print(f"\n  Spearman rank correlation:")
print(f"    rₛ = {r_spear:.4f},  p = {p_spear:.6f}")

# --- Correlation: redlined fraction vs per capita income ---
r_redlined, p_redlined = stats.pearsonr(
    analysis["redlined_fraction"], analysis["per_capita_income"]
)
print(f"\n  Pearson correlation (redlined fraction vs per capita income):")
print(f"    r = {r_redlined:.4f},  p = {p_redlined:.6f}")
if p_redlined < 0.05:
    print(f"    → Statistically significant (p < 0.05)")

# --- Poverty rate correlation ---
pov_analysis = analysis.dropna(subset=["poverty_rate"])
if len(pov_analysis) > 5:
    r_pov, p_pov = stats.pearsonr(
        pov_analysis["weighted_grade_score"], pov_analysis["poverty_rate"]
    )
    print(f"\n  Pearson correlation (weighted HOLC score vs poverty rate):")
    print(f"    r = {r_pov:.4f},  p = {p_pov:.6f}")
    if p_pov < 0.05:
        print(f"    → Statistically significant (p < 0.05)")
    if r_pov > 0:
        print(
            f"    → Positive: worse HOLC grades associated with HIGHER poverty"
        )

# --- Mean income by dominant HOLC grade ---
print(f"\n  Mean per capita income by dominant HOLC grade:")
print(f"  {'Grade':<8} {'Mean $':<12} {'Median $':<12} {'Count':<6}")
print(f"  {'-' * 38}")
for grade in ["A", "B", "C", "D"]:
    subset = analysis[analysis["dominant_grade"] == grade]
    if len(subset) > 0:
        mean_inc = subset["per_capita_income"].mean()
        median_inc = subset["per_capita_income"].median()
        print(
            f"  {grade:<8} ${mean_inc:<11,.0f} ${median_inc:<11,.0f} {len(subset):<6}"
        )

# --- One-way ANOVA ---
groups = [
    analysis[analysis["dominant_grade"] == g]["per_capita_income"].values
    for g in ["A", "B", "C", "D"]
    if len(analysis[analysis["dominant_grade"] == g]) > 0
]
if len(groups) >= 2:
    f_stat, p_anova = stats.f_oneway(*groups)
    print(f"\n  One-way ANOVA (per capita income across HOLC grades):")
    print(f"    F = {f_stat:.4f},  p = {p_anova:.6f}")
    if p_anova < 0.05:
        print(f"    → Significant difference between groups (p < 0.05)")

# --- Point-biserial: redlined (D) vs not ---
analysis["is_redlined"] = (analysis["dominant_grade"] == "D").astype(int)
r_pb, p_pb = stats.pointbiserialr(
    analysis["is_redlined"], analysis["per_capita_income"]
)
print(f"\n  Point-biserial correlation (redlined D vs other grades):")
print(f"    r_pb = {r_pb:.4f},  p = {p_pb:.6f}")
if p_pb < 0.05:
    print(
        f"    → Significant income difference between redlined and non-redlined areas"
    )

# --- Effect size: Cohen's d (Grade A vs Grade D) ---
a_income = analysis[analysis["dominant_grade"] == "A"]["per_capita_income"]
d_income = analysis[analysis["dominant_grade"] == "D"]["per_capita_income"]
if len(a_income) > 1 and len(d_income) > 1:
    pooled_std = np.sqrt(
        (
            (len(a_income) - 1) * a_income.std() ** 2
            + (len(d_income) - 1) * d_income.std() ** 2
        )
        / (len(a_income) + len(d_income) - 2)
    )
    if pooled_std > 0:
        cohens_d = (a_income.mean() - d_income.mean()) / pooled_std
        print(f"\n  Cohen's d (Grade A vs Grade D):")
        print(f"    d = {cohens_d:.4f}")
        if abs(cohens_d) >= 0.8:
            print(f"    → Large effect size")
        elif abs(cohens_d) >= 0.5:
            print(f"    → Medium effect size")
        elif abs(cohens_d) >= 0.2:
            print(f"    → Small effect size")

print("=" * 70)


# ─── 6b. Export statistics to CSV ────────────────────────────────────────────
stats_rows = []

stats_rows.append({
    "Analysis": "Income (Atlanta)",
    "Test": "Pearson (weighted HOLC score vs per capita income)",
    "Statistic": f"r = {r_weighted:.4f}",
    "p-value": f"{p_weighted:.6f}",
})
stats_rows.append({
    "Analysis": "Income (Atlanta)",
    "Test": "Spearman (weighted HOLC score vs per capita income)",
    "Statistic": f"rs = {r_spear:.4f}",
    "p-value": f"{p_spear:.6f}",
})
stats_rows.append({
    "Analysis": "Income (Atlanta)",
    "Test": "Pearson (redlined fraction vs per capita income)",
    "Statistic": f"r = {r_redlined:.4f}",
    "p-value": f"{p_redlined:.6f}",
})
if len(pov_analysis) > 5:
    stats_rows.append({
        "Analysis": "Income (Atlanta)",
        "Test": "Pearson (weighted HOLC score vs poverty rate)",
        "Statistic": f"r = {r_pov:.4f}",
        "p-value": f"{p_pov:.6f}",
    })
if len(groups) >= 2:
    stats_rows.append({
        "Analysis": "Income (Atlanta)",
        "Test": "One-way ANOVA (per capita income across HOLC grades)",
        "Statistic": f"F = {f_stat:.4f}",
        "p-value": f"{p_anova:.6f}",
    })
stats_rows.append({
    "Analysis": "Income (Atlanta)",
    "Test": "Point-biserial (redlined D vs other grades)",
    "Statistic": f"r_pb = {r_pb:.4f}",
    "p-value": f"{p_pb:.6f}",
})
if len(a_income) > 1 and len(d_income) > 1 and pooled_std > 0:
    stats_rows.append({
        "Analysis": "Income (Atlanta)",
        "Test": "Cohen's d (Grade A vs Grade D)",
        "Statistic": f"d = {cohens_d:.4f}",
        "p-value": "N/A",
    })

stats_df = pd.DataFrame(stats_rows)
stats_csv_path = os.path.join(script_dir, "holc_income_statistics.csv")
stats_df.to_csv(stats_csv_path, index=False)
print(f"\n  Statistics saved to {stats_csv_path}")


# ─── 7. Visualization (separate images) ─────────────────────────────────────
print("\nGenerating visualizations...")

# --- Plot 1: Scatter – weighted grade score vs per capita income ---
fig1, ax1 = plt.subplots(figsize=(10, 8))
colors = [COLOR_MAP[g] for g in analysis["dominant_grade"]]
ax1.scatter(
    analysis["weighted_grade_score"],
    analysis["per_capita_income"] / 1000,
    c=colors,
    edgecolors="black",
    s=80,
    alpha=0.8,
    zorder=5,
)

slope, intercept, r, p, se = stats.linregress(
    analysis["weighted_grade_score"], analysis["per_capita_income"] / 1000
)
x_line = np.linspace(1, 4, 100)
ax1.plot(x_line, slope * x_line + intercept, "k--", linewidth=2, alpha=0.7)
ax1.set_xlabel(
    "Area-Weighted HOLC Grade Score\n(1=A/Best → 4=D/Hazardous)", fontsize=11
)
ax1.set_ylabel("Per Capita Income ($K)", fontsize=11)
ax1.set_title(
    f"HOLC Redlining vs Per Capita Income in Atlanta\nr={r:.3f}, p={p:.4f}",
    fontsize=13,
    fontweight="bold",
)
ax1.set_xticks([1, 2, 3, 4])
ax1.set_xticklabels(["1 (A)", "2 (B)", "3 (C)", "4 (D)"])
ax1.grid(True, alpha=0.3)
plt.tight_layout()
out1 = os.path.join(script_dir, "holc_income_scatter.png")
plt.savefig(out1, dpi=300, bbox_inches="tight")
print(f"  Saved {out1}")
plt.show()

# --- Plot 2: Box plot – income by dominant grade ---
fig2, ax2 = plt.subplots(figsize=(10, 5))
grade_data = []
grade_labels = []
grade_colors = []
for g in ["A", "B", "C", "D"]:
    subset = analysis[analysis["dominant_grade"] == g]["per_capita_income"]
    if len(subset) > 0:
        grade_data.append(subset.values / 1000)
        grade_labels.append(f"Grade {g}\n(n={len(subset)})")
        grade_colors.append(COLOR_MAP[g])

bp = ax2.boxplot(grade_data, labels=grade_labels, patch_artist=True, widths=0.6)
for patch, color in zip(bp["boxes"], grade_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax2.set_ylabel("Per Capita Income ($K)", fontsize=11)
ax2.set_title(
    "Income Distribution by Dominant HOLC Grade – Atlanta",
    fontsize=13,
    fontweight="bold",
)
ax2.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
out2 = os.path.join(script_dir, "holc_income_boxplot.png")
plt.savefig(out2, dpi=300, bbox_inches="tight")
print(f"  Saved {out2}")
plt.show()

# --- Plot 3: Scatter – weighted HOLC score vs poverty rate ---
fig3, ax3 = plt.subplots(figsize=(10, 8))
pov_plot = analysis.dropna(subset=["poverty_rate"])
pov_colors = [COLOR_MAP[g] for g in pov_plot["dominant_grade"]]
ax3.scatter(
    pov_plot["weighted_grade_score"],
    pov_plot["poverty_rate"],
    c=pov_colors,
    edgecolors="black",
    s=80,
    alpha=0.8,
    zorder=5,
)
slope3, intercept3, r3, p3, se3 = stats.linregress(
    pov_plot["weighted_grade_score"], pov_plot["poverty_rate"]
)
x_line3 = np.linspace(1, 4, 100)
ax3.plot(x_line3, slope3 * x_line3 + intercept3, "k--", linewidth=2, alpha=0.7)
ax3.set_xlabel(
    "Area-Weighted HOLC Grade Score\n(1=A/Best → 4=D/Hazardous)", fontsize=11
)
ax3.set_ylabel("Poverty Rate (%)", fontsize=11)
ax3.set_title(
    f"HOLC Redlining vs Poverty Rate in Atlanta\nr={r3:.3f}, p={p3:.6f}",
    fontsize=13,
    fontweight="bold",
)
ax3.set_xticks([1, 2, 3, 4])
ax3.set_xticklabels(["1 (A)", "2 (B)", "3 (C)", "4 (D)"])
ax3.grid(True, alpha=0.3)
plt.tight_layout()
out3 = os.path.join(script_dir, "holc_income_poverty_scatter.png")
plt.savefig(out3, dpi=300, bbox_inches="tight")
print(f"  Saved {out3}")
plt.show()

# --- Plot 4: Choropleth map – income with HOLC overlay ---
fig4, ax4 = plt.subplots(figsize=(12, 10))
analysis_geo = analysis.copy()
analysis_geo = analysis_geo.to_crs(epsg=3857)
analysis_geo.plot(
    column="per_capita_income",
    cmap="RdYlGn",
    legend=True,
    ax=ax4,
    edgecolor="gray",
    linewidth=0.5,
    alpha=0.8,
    legend_kwds={
        "label": "Per Capita Income ($)",
        "shrink": 0.7,
        "format": "${x:,.0f}",
    },
)

holc_plot = holc.to_crs(epsg=3857)
for grade in ["D", "C", "B", "A"]:
    subset = holc_plot[holc_plot["grade"] == grade]
    subset.boundary.plot(ax=ax4, color=COLOR_MAP[grade], linewidth=0.4, alpha=0.5)

ax4.set_axis_off()
ax4.set_title(
    "Per Capita Income Map with HOLC Zone Boundaries – Atlanta",
    fontsize=13,
    fontweight="bold",
)

legend_elements = [
    Patch(facecolor=c, edgecolor="black", label=f"HOLC Grade {g}")
    for g, c in COLOR_MAP.items()
]
ax4.legend(handles=legend_elements, loc="lower left", fontsize=9)
plt.tight_layout()
out4 = os.path.join(script_dir, "holc_income_choropleth.png")
plt.savefig(out4, dpi=300, bbox_inches="tight")
print(f"  Saved {out4}")
plt.show()

print("\nDone!")
