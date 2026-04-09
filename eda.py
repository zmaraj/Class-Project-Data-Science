# eda.py - data acquisition, preliminary analysis, EDA, visualization

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# Section 1 - Data Acquisition
print("Data Acquisition & Integrity Check")

# load data w semicolon-delimited
red = pd.read_csv("data/winequality-red.csv", sep=";")
white = pd.read_csv("data/winequality-white.csv", sep=";")

# tag wine type
red["wine_type"] = "red"
white["wine_type"] = "white"

# integrity check
print(f"Red rows (raw): {len(red)}")
print(f"White rows (raw): {len(white)}")
print(f"Missing values: {red.isnull().sum().sum() + white.isnull().sum().sum()}")

# remove duplicates
red_dupes = red.duplicated().sum()
white_dupes = white.duplicated().sum()

red.drop_duplicates(inplace=True)
white.drop_duplicates(inplace=True)

print(f"\nRed duplicates removed: {red_dupes} , {len(red)} rows remain")
print(f"White duplicates removed: {white_dupes} , {len(white)} rows remain")

# combine both datasets into one
df = pd.concat([red, white], ignore_index=True)
print(f"\nCombined dataset shape: {df.shape}")

# binary quality column: >= 7 = good quality (1), < 7 = not good (0)
df["good_quality"] = (df["quality"] >= 7).astype(int)

good = df["good_quality"].sum()
not_good = len(df) - good
print(f"\nBinary quality column added:")
print(f"Good quality (>=7): {good}")
print(f"Not good (<7): {not_good}")

print(f"\nFinal clean dataset shape: {df.shape}")

# final cleaned dataset saved into data folder
df.to_csv("data/wine_cleaned.csv", index=False)
print("Cleaned dataset saved to data/wine_cleaned.csv")


# Section 2 - Preliminary Data Analysis
print("\nPreliminary Data Analysis")

# overview
print(f"\nRows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\nData Types:")
print(df.dtypes)

print("\nVariable Descriptions:")
descriptions = {
    "fixed acidity" : "tartaric acid (g/dm3) - non-volatile acids",
    "volatile acidity" : "acetic acid (g/dm3) - high = vinegar taste",
    "citric acid" : "citric acid (g/dm3) - adds freshness",
    "residual sugar": "sugar left after fermentation (g/dm3)",
    "chlorides" : "salt content (g/dm3)",
    "free sulfur dioxide" : "free SO2 (mg/dm3) - prevents oxidation",
    "total sulfur dioxide" : "total SO2 (mg/dm3) - free + bound",
    "density" : "density (g/cm3)",
    "pH" : "acidity scale - wine typically 3-4",
    "sulphates" : "potassium sulphate (g/dm3) - antioxidant",
    "alcohol" : "alcohol % by volume",
    "quality" : "expert score 0-10",
    "wine_type" : "red or white",
    "good_quality" : "1 = quality >= 7, 0 = quality < 7",
}
for col, desc in descriptions.items():
    print(f"{col:<25}: {desc}")

# missing data
print("\nMissing Values:")
missing = df.isnull().sum()

if missing.sum() == 0:
    print("No missing values found")
else:
    print(missing[missing > 0])

# outlier detection using IQR method
print("\nOutlier Detection (IQR method):")
feature_cols = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar",
                "chlorides", "free sulfur dioxide", "total sulfur dioxide",
                "density", "pH", "sulphates", "alcohol"]

for col in feature_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lo = Q1 - 1.5 * IQR
    hi = Q3 + 1.5 * IQR
    n = ((df[col] < lo) | (df[col] > hi)).sum()
    print(f"{col:<25}: {n:>4} outliers (bounds: {lo:.3f} to {hi:.3f})")

print("\nOutliers kept - extreme chemical values are natural in wine data")


# Section 3 - Exploratory Data Analysis
print("\nExploratory Data Analysis")

# summary statistics
print("\nSummary Statistics:")
print(df[feature_cols + ["quality"]].agg(["mean", "median", "std", "min", "max"]).round(3).to_string())

# mode
print("\nMode (most frequent value):")
for col in feature_cols + ["quality"]:
    print(f"{col:<25}: {df[col].mode()[0]}")

# mean values by wine type
print("\nMean Values by Wine Type:")
print(df.groupby("wine_type")[feature_cols + ["quality"]].mean().round(3).T.to_string())

# correlation with quality score
print("\nPearson Correlation with Quality Score (sorted by strength):")
corr = df[feature_cols + ["quality"]].corr()["quality"].drop("quality")
print(corr.sort_values(key=abs, ascending=False).round(3).to_string())

# key findings
print("\nKey Findings:")
print("- Alcohol has the strongest positive correlation with quality")
print("- Volatile acidity has the strongest negative correlation (vinegar effect)")
print("- White wines have much higher total sulfur dioxide")
print("- Only ~22% of wines score >= 7 (good quality) - dataset is imbalanced")
print("- Red wine max quality = 8, white wine max quality = 9")


# Section 4 - Data Visualization
print("\nData Visualization")

RED_COLOR = "#c0392b"
WHITE_COLOR = "#2980b9"
PALETTE = {"red": RED_COLOR, "white": WHITE_COLOR}
plt.style.use("seaborn-v0_8-whitegrid")

# fig 1 - quality score bar chart
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Fig 1 - Quality Score Distribution", fontweight="bold")

for ax, wtype, color in zip(axes, ["red", "white"], [RED_COLOR, WHITE_COLOR]):
    sub = df[df["wine_type"] == wtype]
    counts = sub["quality"].value_counts().sort_index()
    ax.bar(counts.index, counts.values, color=color, edgecolor="white", alpha=0.85)
    ax.set_title(f"{wtype.capitalize()} Wine (n={len(sub)})")
    ax.set_xlabel("Quality Score")
    ax.set_ylabel("Count")
    ax.set_xticks(range(3, 10))
    for x, y in zip(counts.index, counts.values):
        ax.text(x, y + 2, str(y), ha="center", fontsize=9)

plt.tight_layout()
plt.savefig("fig1_quality_distribution.png", dpi=150)
plt.show()

# fig 2 - good vs not-good binary quality bar chart
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle("Fig 2 - Good Quality (>=7) vs Not Good (<7)", fontweight="bold")

for ax, wtype, color in zip(axes, ["red", "white"], [RED_COLOR, WHITE_COLOR]):
    sub = df[df["wine_type"] == wtype]
    counts = sub["good_quality"].value_counts().sort_index()
    ax.bar(["Not Good (0)", "Good (1)"], counts.values,
           color=[color, "#27ae60"], edgecolor="white", alpha=0.85)
    ax.set_title(f"{wtype.capitalize()} Wine")
    ax.set_ylabel("Count")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 2, str(v), ha="center", fontsize=10)

plt.tight_layout()
plt.savefig("fig2_binary_quality.png", dpi=150)
plt.show()

# fig 3 - histograms of all features
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
fig.suptitle("Fig 3 - Feature Distributions (Red vs White)", fontweight="bold")
axes = axes.flatten()

for i, col in enumerate(feature_cols):
    for wtype, color in PALETTE.items():
        axes[i].hist(df[df["wine_type"] == wtype][col],
                     bins=30, alpha=0.55, color=color, label=wtype)
    axes[i].set_title(col, fontsize=9)
    axes[i].legend(fontsize=7)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig("fig3_histograms.png", dpi=150)
plt.show()

# fig 4 - box plots: key features vs quality score
key_features = ["alcohol", "volatile acidity", "citric acid",
                "sulphates", "residual sugar", "density"]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Fig 4 - Key Features vs Quality Score", fontweight="bold")
axes = axes.flatten()

for i, col in enumerate(key_features):
    sns.boxplot(data=df, x="quality", y=col, hue="wine_type",
                palette=PALETTE, ax=axes[i], linewidth=0.8, fliersize=2)
    axes[i].set_title(col)
    axes[i].set_xlabel("Quality Score")
    axes[i].legend(fontsize=7)

plt.tight_layout()
plt.savefig("fig4_boxplots.png", dpi=150)
plt.show()

# fig 5 - correlation heatmaps
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Fig 5 - Correlation Heatmap (Red vs White)", fontweight="bold")

for ax, wtype, color in zip(axes, ["red", "white"], [RED_COLOR, WHITE_COLOR]):
    sub = df[df["wine_type"] == wtype][feature_cols + ["quality"]]
    corr = sub.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, ax=ax, annot=True, fmt=".2f",
                cmap="RdBu_r", center=0, linewidths=0.5,
                annot_kws={"size": 7}, vmin=-1, vmax=1)
    ax.set_title(f"{wtype.capitalize()} Wine", color=color, fontweight="bold")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=8)

plt.tight_layout()
plt.savefig("fig5_heatmap.png", dpi=150)
plt.show()

# fig 6 - scatter plots: alcohol and volatile acidity vs quality
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Fig 6 - Top Correlates vs Quality", fontweight="bold")

for wtype, color in PALETTE.items():
    sub = df[df["wine_type"] == wtype]
    jitter = np.random.uniform(-0.2, 0.2, size=len(sub))
    axes[0].scatter(sub["alcohol"], sub["quality"] + jitter,
                    color=color, alpha=0.2, s=10, label=wtype)
    axes[1].scatter(sub["volatile acidity"], sub["quality"] + jitter,
                    color=color, alpha=0.2, s=10, label=wtype)

for ax, xcol, xlabel in zip(axes,
                             ["alcohol", "volatile acidity"],
                             ["Alcohol (%)", "Volatile Acidity (g/dm3)"]):
    m, b, *_ = stats.linregress(df[xcol], df["quality"])
    x_line = np.linspace(df[xcol].min(), df[xcol].max(), 100)
    ax.plot(x_line, m * x_line + b, "k--", linewidth=1.5, label="trend")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Quality Score (jittered)")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("fig6_scatter.png", dpi=150)
plt.show()

print("\nAll 6 figures saved as PNG files.")