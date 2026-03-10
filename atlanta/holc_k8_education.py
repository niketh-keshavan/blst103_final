"""
HOLC Redlining and Education Correlation Analysis – Atlanta, Georgia
====================================================================
This script analyzes the correlation between 1930s HOLC redlining grades
in Atlanta and current educational outcomes by census tract.

Because Atlanta does not have a single city-level school data portal like
Chicago's CPS, this script uses Census Bureau ACS data on educational
attainment for the population 25+. The key metrics are:
  - % of adults with a bachelor's degree or higher
  - % of adults without a high school diploma

These census-tract-level measures capture the long-term educational
impact of HOLC redlining on community outcomes.

Data sources:
- HOLC redlining zones: Mapping Inequality (University of Richmond)
- Educational attainment: U.S. Census Bureau ACS 5-Year Estimates (B15003)
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

# ACS Table B15003 — Educational Attainment for the Population 25 and Over
# We request the total plus each level from HS diploma through doctorate.
# B15003_001E = Total
# B15003_017E = Regular high school diploma
# B15003_018E = GED or alternative credential
# B15003_019E = Some college, less than 1 year
# B15003_020E = Some college, 1 or more years, no degree
# B15003_021E = Associate's degree
# B15003_022E = Bachelor's degree
# B15003_023E = Master's degree
# B15003_024E = Professional school degree
# B15003_025E = Doctorate degree
EDU_VARIABLES = (
    "NAME,"
    "B15003_001E,"
    "B15003_017E,B15003_018E,B15003_019E,B15003_020E,B15003_021E,"
    "B15003_022E,B15003_023E,B15003_024E,B15003_025E"
)


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
            raise RuntimeError("Could not download Atlanta HOLC data.")
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


def download_acs_education():
    """Download educational attainment data from Census ACS API."""
    all_rows = []
    header = None

    for county in ATLANTA_COUNTIES:
        url = (
            f"https://api.census.gov/data/2022/acs/acs5"
            f"?get={EDU_VARIABLES}"
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
        raise RuntimeError("No ACS education data retrieved")

    df = pd.DataFrame(all_rows, columns=header)

    # Construct GEOID
    df["GEOID"] = df["state"] + df["county"] + df["tract"]

    # Parse all numeric columns
    for col in header:
        if col.startswith("B15003"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Compute derived metrics
    df["total_25plus"] = df["B15003_001E"]

    # Sum of HS diploma and above (B15003_017E through B15003_025E)
    hs_and_above_cols = [
        "B15003_017E", "B15003_018E", "B15003_019E", "B15003_020E",
        "B15003_021E", "B15003_022E", "B15003_023E", "B15003_024E",
        "B15003_025E",
    ]
    df["hs_or_more"] = df[hs_and_above_cols].sum(axis=1)

    # Bachelor's degree or higher (B15003_022E through B15003_025E)
    bachelors_plus_cols = ["B15003_022E", "B15003_023E", "B15003_024E", "B15003_025E"]
    df["bachelors_plus"] = df[bachelors_plus_cols].sum(axis=1)

    # Percentages
    df["pct_no_hs_diploma"] = np.where(
        df["total_25plus"] > 0,
        (df["total_25plus"] - df["hs_or_more"]) / df["total_25plus"] * 100,
        np.nan,
    )
    df["pct_bachelors_plus"] = np.where(
        df["total_25plus"] > 0,
        df["bachelors_plus"] / df["total_25plus"] * 100,
        np.nan,
    )

    df = df.dropna(subset=["pct_bachelors_plus", "pct_no_hs_diploma"])
    df = df[df["total_25plus"] > 0]

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


# ─── 3. Download educational attainment data ────────────────────────────────
print("\nDownloading educational attainment data from Census ACS 5-Year...")
edu_data = download_acs_education()
print(f"  Education data for {len(edu_data)} census tracts")
print(
    f"  % Bachelor's or higher: "
    f"{edu_data['pct_bachelors_plus'].min():.1f}% – "
    f"{edu_data['pct_bachelors_plus'].max():.1f}%"
)
print(
    f"  % No HS diploma: "
    f"{edu_data['pct_no_hs_diploma'].min():.1f}% – "
    f"{edu_data['pct_no_hs_diploma'].max():.1f}%"
)


# ─── 4. Merge tracts with education data ────────────────────────────────────
print("\nMerging census tracts with education data...")
tracts = tracts.merge(
    edu_data[["GEOID", "pct_bachelors_plus", "pct_no_hs_diploma", "total_25plus"]],
    on="GEOID",
    how="left",
)
tracts = tracts.dropna(subset=["pct_bachelors_plus"])
print(f"  {len(tracts)} tracts with valid education data")


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

analysis = tracts.merge(results_df, on="GEOID", how="inner")
analysis = analysis.dropna(subset=["pct_bachelors_plus", "weighted_grade_score"])
print(f"  Final analysis dataset: {len(analysis)} census tracts")


# ─── 6. Statistical analysis ────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  STATISTICAL RESULTS — HOLC Grade vs Educational Attainment (Atlanta)")
print("=" * 70)

# --- Pearson: weighted HOLC score vs % bachelor's+ ---
r_bach, p_bach = stats.pearsonr(
    analysis["weighted_grade_score"], analysis["pct_bachelors_plus"]
)
print(f"\n  Pearson correlation (weighted HOLC score vs % bachelor's+):")
print(f"    r = {r_bach:.4f},  p = {p_bach:.6f}")
if p_bach < 0.05:
    print(f"    → Statistically significant (p < 0.05)")
if r_bach < 0:
    print(
        f"    → Negative: worse HOLC grades → LOWER college attainment"
    )

# --- Spearman ---
r_spear, p_spear = stats.spearmanr(
    analysis["weighted_grade_score"], analysis["pct_bachelors_plus"]
)
print(f"\n  Spearman rank correlation:")
print(f"    rₛ = {r_spear:.4f},  p = {p_spear:.6f}")

# --- Pearson: weighted HOLC score vs % no HS diploma ---
r_nohs, p_nohs = stats.pearsonr(
    analysis["weighted_grade_score"], analysis["pct_no_hs_diploma"]
)
print(f"\n  Pearson correlation (weighted HOLC score vs % no HS diploma):")
print(f"    r = {r_nohs:.4f},  p = {p_nohs:.6f}")
if p_nohs < 0.05:
    print(f"    → Statistically significant (p < 0.05)")
if r_nohs > 0:
    print(
        f"    → Positive: worse HOLC grades → MORE adults without HS diploma"
    )

# --- Redlined fraction vs % bachelor's+ ---
r_red, p_red = stats.pearsonr(
    analysis["redlined_fraction"], analysis["pct_bachelors_plus"]
)
print(f"\n  Pearson correlation (redlined fraction vs % bachelor's+):")
print(f"    r = {r_red:.4f},  p = {p_red:.6f}")
if p_red < 0.05:
    print(f"    → Statistically significant (p < 0.05)")

# --- Mean education by dominant HOLC grade ---
print(f"\n  Mean educational attainment by dominant HOLC grade:")
print(f"  {'Grade':<8} {'% Bach+':<10} {'% No HS':<10} {'Count':<6}")
print(f"  {'-' * 34}")
for grade in ["A", "B", "C", "D"]:
    subset = analysis[analysis["dominant_grade"] == grade]
    if len(subset) > 0:
        mean_bach = subset["pct_bachelors_plus"].mean()
        mean_nohs = subset["pct_no_hs_diploma"].mean()
        print(
            f"  {grade:<8} {mean_bach:<10.1f} {mean_nohs:<10.1f} {len(subset):<6}"
        )

# --- One-way ANOVA ---
groups = [
    analysis[analysis["dominant_grade"] == g]["pct_bachelors_plus"].values
    for g in ["A", "B", "C", "D"]
    if len(analysis[analysis["dominant_grade"] == g]) > 0
]
if len(groups) >= 2:
    f_stat, p_anova = stats.f_oneway(*groups)
    print(f"\n  One-way ANOVA (% bachelor's+ across HOLC grades):")
    print(f"    F = {f_stat:.4f},  p = {p_anova:.6f}")
    if p_anova < 0.05:
        print(f"    → Significant difference between groups (p < 0.05)")

# --- Point-biserial: redlined (D) vs not ---
analysis["is_redlined"] = (analysis["dominant_grade"] == "D").astype(int)
r_pb, p_pb = stats.pointbiserialr(
    analysis["is_redlined"], analysis["pct_bachelors_plus"]
)
print(f"\n  Point-biserial correlation (redlined D vs other grades):")
print(f"    r_pb = {r_pb:.4f},  p = {p_pb:.6f}")
if p_pb < 0.05:
    print(
        f"    → Significant education gap between redlined and non-redlined areas"
    )

# --- Cohen's d (Grade A vs Grade D) ---
a_edu = analysis[analysis["dominant_grade"] == "A"]["pct_bachelors_plus"]
d_edu = analysis[analysis["dominant_grade"] == "D"]["pct_bachelors_plus"]
if len(a_edu) > 1 and len(d_edu) > 1:
    pooled_std = np.sqrt(
        (
            (len(a_edu) - 1) * a_edu.std() ** 2
            + (len(d_edu) - 1) * d_edu.std() ** 2
        )
        / (len(a_edu) + len(d_edu) - 2)
    )
    if pooled_std > 0:
        cohens_d = (a_edu.mean() - d_edu.mean()) / pooled_std
        print(f"\n  Cohen's d (Grade A vs Grade D, % bachelor's+):")
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
    "Analysis": "Education (Atlanta)",
    "Test": "Pearson (weighted HOLC score vs % bachelor's+)",
    "Statistic": f"r = {r_bach:.4f}",
    "p-value": f"{p_bach:.6f}",
})
stats_rows.append({
    "Analysis": "Education (Atlanta)",
    "Test": "Spearman (weighted HOLC score vs % bachelor's+)",
    "Statistic": f"rs = {r_spear:.4f}",
    "p-value": f"{p_spear:.6f}",
})
stats_rows.append({
    "Analysis": "Education (Atlanta)",
    "Test": "Pearson (weighted HOLC score vs % no HS diploma)",
    "Statistic": f"r = {r_nohs:.4f}",
    "p-value": f"{p_nohs:.6f}",
})
stats_rows.append({
    "Analysis": "Education (Atlanta)",
    "Test": "Pearson (redlined fraction vs % bachelor's+)",
    "Statistic": f"r = {r_red:.4f}",
    "p-value": f"{p_red:.6f}",
})
if len(groups) >= 2:
    stats_rows.append({
        "Analysis": "Education (Atlanta)",
        "Test": "One-way ANOVA (% bachelor's+ across HOLC grades)",
        "Statistic": f"F = {f_stat:.4f}",
        "p-value": f"{p_anova:.6f}",
    })
stats_rows.append({
    "Analysis": "Education (Atlanta)",
    "Test": "Point-biserial (redlined D vs other grades)",
    "Statistic": f"r_pb = {r_pb:.4f}",
    "p-value": f"{p_pb:.6f}",
})
if len(a_edu) > 1 and len(d_edu) > 1 and pooled_std > 0:
    stats_rows.append({
        "Analysis": "Education (Atlanta)",
        "Test": "Cohen's d (Grade A vs Grade D, % bachelor's+)",
        "Statistic": f"d = {cohens_d:.4f}",
        "p-value": "N/A",
    })

stats_df = pd.DataFrame(stats_rows)
stats_csv_path = os.path.join(script_dir, "holc_education_statistics.csv")
stats_df.to_csv(stats_csv_path, index=False)
print(f"\n  Statistics saved to {stats_csv_path}")


# ─── 7. Visualization (separate images) ─────────────────────────────────────
print("\nGenerating visualizations...")

# --- Plot 1: Scatter – HOLC score vs % bachelor's+ ---
fig1, ax1 = plt.subplots(figsize=(10, 8))
colors = [COLOR_MAP[g] for g in analysis["dominant_grade"]]
ax1.scatter(
    analysis["weighted_grade_score"],
    analysis["pct_bachelors_plus"],
    c=colors,
    edgecolors="black",
    s=80,
    alpha=0.8,
    zorder=5,
)
slope, intercept, r, p, se = stats.linregress(
    analysis["weighted_grade_score"], analysis["pct_bachelors_plus"]
)
x_line = np.linspace(1, 4, 100)
ax1.plot(x_line, slope * x_line + intercept, "k--", linewidth=2, alpha=0.7)
ax1.set_xlabel(
    "Area-Weighted HOLC Grade Score\n(1=A/Best → 4=D/Hazardous)", fontsize=11
)
ax1.set_ylabel("% Adults with Bachelor's Degree or Higher", fontsize=11)
ax1.set_title(
    f"HOLC Score vs College Attainment in Atlanta\nr={r:.3f}, p={p:.4f}",
    fontsize=13,
    fontweight="bold",
)
ax1.set_xticks([1, 2, 3, 4])
ax1.set_xticklabels(["1 (A)", "2 (B)", "3 (C)", "4 (D)"])
ax1.grid(True, alpha=0.3)
plt.tight_layout()
out1 = os.path.join(script_dir, "holc_education_scatter_bachelors.png")
plt.savefig(out1, dpi=300, bbox_inches="tight")
print(f"  Saved {out1}")
plt.show()

# --- Plot 2: Box plot – % bachelor's+ by dominant grade ---
fig2, ax2 = plt.subplots(figsize=(10, 5))
grade_data = []
grade_labels = []
grade_colors = []
for g in ["A", "B", "C", "D"]:
    subset = analysis[analysis["dominant_grade"] == g]["pct_bachelors_plus"]
    if len(subset) > 0:
        grade_data.append(subset.values)
        grade_labels.append(f"Grade {g}\n(n={len(subset)})")
        grade_colors.append(COLOR_MAP[g])

bp = ax2.boxplot(grade_data, labels=grade_labels, patch_artist=True, widths=0.6)
for patch, color in zip(bp["boxes"], grade_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax2.set_ylabel("% Adults with Bachelor's Degree or Higher", fontsize=11)
ax2.set_title(
    "College Attainment by Dominant HOLC Grade – Atlanta",
    fontsize=13,
    fontweight="bold",
)
ax2.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
out2 = os.path.join(script_dir, "holc_education_boxplot.png")
plt.savefig(out2, dpi=300, bbox_inches="tight")
print(f"  Saved {out2}")
plt.show()

# --- Plot 3: Scatter – HOLC score vs % no HS diploma ---
fig3, ax3 = plt.subplots(figsize=(10, 8))
nohs_colors = [COLOR_MAP[g] for g in analysis["dominant_grade"]]
ax3.scatter(
    analysis["weighted_grade_score"],
    analysis["pct_no_hs_diploma"],
    c=nohs_colors,
    edgecolors="black",
    s=80,
    alpha=0.8,
    zorder=5,
)
slope3, intercept3, r3, p3, se3 = stats.linregress(
    analysis["weighted_grade_score"], analysis["pct_no_hs_diploma"]
)
x_line3 = np.linspace(1, 4, 100)
ax3.plot(x_line3, slope3 * x_line3 + intercept3, "k--", linewidth=2, alpha=0.7)
ax3.set_xlabel(
    "Area-Weighted HOLC Grade Score\n(1=A/Best → 4=D/Hazardous)", fontsize=11
)
ax3.set_ylabel("% Adults Without HS Diploma", fontsize=11)
ax3.set_title(
    f"HOLC Score vs % No High School Diploma in Atlanta\nr={r3:.3f}, p={p3:.6f}",
    fontsize=13,
    fontweight="bold",
)
ax3.set_xticks([1, 2, 3, 4])
ax3.set_xticklabels(["1 (A)", "2 (B)", "3 (C)", "4 (D)"])
ax3.grid(True, alpha=0.3)
plt.tight_layout()
out3 = os.path.join(script_dir, "holc_education_scatter_nohs.png")
plt.savefig(out3, dpi=300, bbox_inches="tight")
print(f"  Saved {out3}")
plt.show()

# --- Plot 4: Choropleth map – education with HOLC overlay ---
fig4, ax4 = plt.subplots(figsize=(12, 10))
analysis_geo = analysis.copy()
analysis_geo = analysis_geo.to_crs(epsg=3857)
analysis_geo.plot(
    column="pct_bachelors_plus",
    cmap="RdYlGn",
    legend=True,
    ax=ax4,
    edgecolor="gray",
    linewidth=0.5,
    alpha=0.8,
    legend_kwds={
        "label": "% Bachelor's Degree or Higher",
        "shrink": 0.7,
    },
)

holc_plot = holc.to_crs(epsg=3857)
for grade in ["D", "C", "B", "A"]:
    subset = holc_plot[holc_plot["grade"] == grade]
    subset.boundary.plot(ax=ax4, color=COLOR_MAP[grade], linewidth=0.4, alpha=0.5)

ax4.set_axis_off()
ax4.set_title(
    "Educational Attainment Map with HOLC Zone Boundaries – Atlanta",
    fontsize=13,
    fontweight="bold",
)

legend_elements = [
    Patch(facecolor=c, edgecolor="black", label=f"HOLC Grade {g}")
    for g, c in COLOR_MAP.items()
]
ax4.legend(handles=legend_elements, loc="lower left", fontsize=9)
plt.tight_layout()
out4 = os.path.join(script_dir, "holc_education_choropleth.png")
plt.savefig(out4, dpi=300, bbox_inches="tight")
print(f"  Saved {out4}")
plt.show()

print("\nDone!")
