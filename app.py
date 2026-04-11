# app.py - Streamlit Web App
# Wine Quality Analysis - Integrates EDA, Hypothesis Testing & ML

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="Wine Quality Analysis",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;600;700&family=Montserrat:wght@300;400;500;600&display=swap');

.stApp { background-color: #ffffff; }

header[data-testid="stHeader"] { background-color: #ffffff !important; }
[data-testid="stSidebarCollapseButton"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebarCollapsedControl"] { display: none !important; }
button[data-testid="baseButton-headerNoPadding"] { display: none !important; }

/* All text default */
body, p, li, span, div {
    font-family: 'Montserrat', sans-serif !important;
}

h1 {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 3rem !important;
    font-weight: 700 !important;
    color: #6B0F1A !important;
    letter-spacing: 1px;
    margin-bottom: 0.1rem !important;
}

h2 {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 1.6rem !important;
    color: #3a0a10 !important;
    font-weight: 600 !important;
    border-left: 3px solid #6B0F1A;
    padding-left: 12px;
    margin-top: 1.5rem !important;
}

h3 {
    font-family: 'Montserrat', sans-serif !important;
    color: #6B0F1A !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    text-transform: uppercase;
    letter-spacing: 2px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #fdf8f5 !important;
    border-right: 1px solid #e8ddd8 !important;
}

[data-testid="stSidebar"] * {
    font-family: 'Montserrat', sans-serif !important;
}

/* Nav radio label — the MENU label */
[data-testid="stSidebar"] .stRadio > label {
    font-family: 'Montserrat', sans-serif !important;
    font-size: 0.65rem !important;
    font-weight: 600 !important;
    color: #b09090 !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
}

/* Nav items */
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    color: #6B0F1A !important;
    letter-spacing: 0.5px !important;
    padding: 5px 0 !important;
    text-transform: none !important;
}

[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    color: #3a0a10 !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e8ddd8;
    border-radius: 10px;
    padding: 1rem 1.2rem !important;
    box-shadow: 0 1px 4px rgba(107,15,26,0.06);
}

[data-testid="stMetricLabel"] {
    font-family: 'Montserrat', sans-serif !important;
    font-size: 0.7rem !important;
    color: #a09090 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.2px !important;
}

[data-testid="stMetricValue"] {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 2.2rem !important;
    color: #6B0F1A !important;
    font-weight: 700 !important;
}

hr { border-color: #e8ddd8 !important; }

.stAlert {
    border-radius: 8px !important;
    font-family: 'Montserrat', sans-serif !important;
    font-size: 0.85rem !important;
}

.stSelectbox label, .stSlider label {
    font-family: 'Montserrat', sans-serif !important;
    font-size: 0.72rem !important;
    color: #a09090 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}

.page-subtitle {
    font-family: 'Montserrat', sans-serif;
    font-size: 0.78rem;
    color: #b09090;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 1.8rem;
}

.section-tag {
    display: inline-block;
    background: rgba(107, 15, 26, 0.07);
    color: #6B0F1A;
    font-family: 'Montserrat', sans-serif;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    padding: 4px 14px;
    border-radius: 20px;
    border: 1px solid rgba(107, 15, 26, 0.2);
    margin-bottom: 0.5rem;
}

.stButton > button {
    background: #6B0F1A !important;
    color: #ffffff !important;
    font-family: 'Montserrat', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.65rem 2rem !important;
}

.stButton > button:hover { opacity: 0.85 !important; }
</style>
""", unsafe_allow_html=True)

# constants
RED     = "#c0392b"
BLUE    = "#2980b9"
WINE    = "#6B0F1A"
PALETTE = {"red": RED, "white": BLUE}

FEATURE_COLS = [
    "fixed acidity", "volatile acidity", "citric acid",
    "residual sugar", "chlorides", "free sulfur dioxide",
    "total sulfur dioxide", "density", "pH", "sulphates",
    "alcohol", "wine_type_encoded"
]

# data + model
@st.cache_data
def load_data():
    df = pd.read_csv("data/wine_cleaned.csv")
    le = LabelEncoder()
    df["wine_type_encoded"] = le.fit_transform(df["wine_type"])
    return df, le

@st.cache_resource
def train_model(df):
    X = df[FEATURE_COLS]
    y = df["good_quality"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10,
        random_state=42, class_weight="balanced"
    )
    rf.fit(X_train, y_train)
    return rf, X_test, y_test

df, le = load_data()
rf, X_test, y_test = train_model(df)
y_pred = rf.predict(X_test)

# sidebar
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:2rem 0 1.2rem 0; border-bottom:1px solid #e8ddd8; margin-bottom:1.8rem;'>
        <div style='font-size:1.8rem; margin-bottom:8px;'>🍷</div>
        <div style='font-family:"Cormorant Garamond",serif; font-size:1.5rem; font-weight:700;
                    color:#6B0F1A; letter-spacing:1px;'>Vino Insight</div>
        <div style='font-family:Montserrat,sans-serif; font-size:0.62rem; color:#c0a8a8;
                    letter-spacing:3px; text-transform:uppercase; margin-top:4px;'>Wine Quality Analysis</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "MENU",
        [
            "Overview",
            "Exploratory Analysis",
            "Hypothesis Testing",
            "Predictive Modeling",
            "Quality Predictor"
        ]
    )

    st.markdown("---")
    st.markdown(f"""
    <div style='font-family:Montserrat,sans-serif; font-size:0.72rem; color:#c0a8a8; line-height:2.2;'>
        <span style='color:#6B0F1A; font-size:0.6rem;'>&#9679;</span>&nbsp; Dataset &nbsp;&nbsp; UCI Wine Quality<br>
        <span style='color:#6B0F1A; font-size:0.6rem;'>&#9679;</span>&nbsp; Wines &nbsp;&nbsp;&nbsp;&nbsp; {len(df):,} records<br>
        <span style='color:#6B0F1A; font-size:0.6rem;'>&#9679;</span>&nbsp; Features &nbsp; 12 chemical<br>
        <span style='color:#6B0F1A; font-size:0.6rem;'>&#9679;</span>&nbsp; Model &nbsp;&nbsp;&nbsp;&nbsp; Random Forest
    </div>
    <br>
    <div style='font-family:Montserrat,sans-serif; font-size:0.62rem; color:#d8c8c8;
                text-align:center; letter-spacing:2px; text-transform:uppercase;'>
        Momin &nbsp;·&nbsp; Zara &nbsp;·&nbsp; Sidney
    </div>
    """, unsafe_allow_html=True)

# chart helpers
def light_fig(w=11, h=4.5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fdf8f5")
    ax.tick_params(colors="#777777", labelsize=9)
    ax.xaxis.label.set_color("#777777")
    ax.yaxis.label.set_color("#777777")
    ax.title.set_color(WINE)
    for spine in ax.spines.values():
        spine.set_edgecolor("#e8ddd8")
    return fig, ax

def light_figs(rows, cols, w=13, h=5):
    fig, axes = plt.subplots(rows, cols, figsize=(w, h))
    fig.patch.set_facecolor("#ffffff")
    for ax in (axes.flatten() if hasattr(axes, 'flatten') else [axes]):
        ax.set_facecolor("#fdf8f5")
        ax.tick_params(colors="#777777", labelsize=9)
        ax.xaxis.label.set_color("#777777")
        ax.yaxis.label.set_color("#777777")
        ax.title.set_color(WINE)
        for spine in ax.spines.values():
            spine.set_edgecolor("#e8ddd8")
    return fig, axes


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Overview
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown('<div class="section-tag">UCI Machine Learning Repository</div>', unsafe_allow_html=True)
    st.title("Wine Quality Analysis")
    st.markdown('<p class="page-subtitle">A data science deep-dive into what makes a wine great</p>', unsafe_allow_html=True)
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Wines",  f"{len(df):,}")
    col2.metric("Red Wines",    f"{(df['wine_type']=='red').sum():,}")
    col3.metric("White Wines",  f"{(df['wine_type']=='white').sum():,}")
    col4.metric("Features",     "12")

    st.markdown("---")
    st.subheader("Dataset Sample")
    wine_filter = st.selectbox("Filter by wine type", ["All", "Red", "White"])
    if wine_filter == "Red":
        display_df = df[df["wine_type"] == "red"]
    elif wine_filter == "White":
        display_df = df[df["wine_type"] == "white"]
    else:
        display_df = df
    st.dataframe(display_df.drop(columns=["wine_type_encoded"]).head(50), use_container_width=True)

    st.markdown("---")
    st.subheader("Summary Statistics")
    st.dataframe(df[FEATURE_COLS[:-1]].describe().round(3), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Exploratory Analysis":
    st.markdown('<div class="section-tag">Zara</div>', unsafe_allow_html=True)
    st.title("Exploratory Analysis")
    st.markdown('<p class="page-subtitle">Distributions, relationships & visual patterns across the dataset</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.subheader("Quality Score Distribution")
    fig, axes = light_figs(1, 2, w=12, h=4)
    for ax, wtype, color in zip(axes, ["red", "white"], [RED, BLUE]):
        sub = df[df["wine_type"] == wtype]
        counts = sub["quality"].value_counts().sort_index()
        ax.bar(counts.index, counts.values, color=color, edgecolor="#ffffff", alpha=0.9, width=0.6)
        ax.set_title(f"{wtype.capitalize()} Wine  (n={len(sub):,})", fontsize=11)
        ax.set_xlabel("Quality Score")
        ax.set_ylabel("Count")
        ax.set_xticks(range(3, 10))
        for x, y_val in zip(counts.index, counts.values):
            ax.text(x, y_val + 5, str(y_val), ha="center", fontsize=8, color="#777777")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("Feature Distribution by Wine Type")
    selected_feature = st.selectbox("Select a feature", FEATURE_COLS[:-1])
    fig, ax = light_fig(9, 4)
    for wtype, color in PALETTE.items():
        ax.hist(df[df["wine_type"] == wtype][selected_feature],
                bins=30, alpha=0.65, color=color, label=wtype.capitalize(), edgecolor="#ffffff")
    ax.set_xlabel(selected_feature)
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of {selected_feature.title()}")
    ax.legend(facecolor="#ffffff", edgecolor="#e8ddd8", labelcolor="#555555")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("Feature vs Quality Score")
    box_feature = st.selectbox("Select feature for box plot",
                                ["alcohol", "volatile acidity", "citric acid",
                                 "sulphates", "residual sugar", "density"])
    fig, ax = light_fig(11, 4.5)
    sns.boxplot(data=df, x="quality", y=box_feature, hue="wine_type",
                palette=PALETTE, ax=ax, linewidth=0.8, fliersize=2)
    ax.set_facecolor("#fdf8f5")
    ax.set_title(f"{box_feature.title()} by Quality Score")
    ax.set_xlabel("Quality Score")
    legend = ax.get_legend()
    if legend:
        legend.get_frame().set_facecolor("#ffffff")
        legend.get_frame().set_edgecolor("#e8ddd8")
        for text in legend.get_texts():
            text.set_color("#555555")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("Correlation Heatmap")
    heatmap_type = st.radio("Wine type", ["red", "white"], horizontal=True)
    sub = df[df["wine_type"] == heatmap_type][FEATURE_COLS[:-1] + ["quality"]]
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    mask = np.triu(np.ones_like(sub.corr(), dtype=bool))
    sns.heatmap(sub.corr(), mask=mask, ax=ax, annot=True, fmt=".2f",
                cmap="RdBu_r", center=0, linewidths=0.4,
                annot_kws={"size": 7.5}, vmin=-1, vmax=1, linecolor="#ffffff")
    ax.set_title(f"{heatmap_type.capitalize()} Wine — Correlation Matrix", color=WINE, fontsize=12)
    ax.tick_params(colors="#777777", labelsize=8)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("Key Findings")
    col1, col2 = st.columns(2)
    with col1:
        st.info("🍷 **Alcohol** has the strongest positive correlation with quality.")
        st.info("⚗️ White wines have significantly higher total sulfur dioxide.")
    with col2:
        st.info("🍶 **Volatile acidity** has the strongest negative correlation.")
        st.info("⚠️ Only ~22% of wines score ≥ 7 so the dataset is class-imbalanced.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Hypothesis Testing
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Hypothesis Testing":
    st.markdown('<div class="section-tag">Sidney</div>', unsafe_allow_html=True)
    st.title("Hypothesis Testing")
    st.markdown('<p class="page-subtitle">Two-sample t-test — alcohol content across wine types</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.subheader("Research Question")
    st.markdown("> **Is there a statistically significant difference in mean alcohol content between red wine and white wine?**")

    st.subheader("Hypotheses")
    col1, col2 = st.columns(2)
    col1.markdown("**H₀ (Null):** There is *no* significant difference in mean alcohol content between red and white wines.")
    col2.markdown("**Hₐ (Alternative):** There *is* a significant difference in mean alcohol content between red and white wines.")

    st.markdown("---")
    alpha = st.slider("Significance level (α)", 0.01, 0.10, 0.05, 0.01)

    red_alcohol   = df[df["wine_type"] == "red"]["alcohol"]
    white_alcohol = df[df["wine_type"] == "white"]["alcohol"]

    st.subheader("Assumption Checks")
    col1, col2 = st.columns(2)
    col1.metric("Red Wine — Mean Alcohol",   f"{red_alcohol.mean():.3f}%")
    col1.metric("Red Wine — Std Dev",         f"{red_alcohol.std():.3f}")
    col2.metric("White Wine — Mean Alcohol", f"{white_alcohol.mean():.3f}%")
    col2.metric("White Wine — Std Dev",       f"{white_alcohol.std():.3f}")

    _, p_red   = stats.shapiro(red_alcohol.sample(min(500, len(red_alcohol)), random_state=42))
    _, p_white = stats.shapiro(white_alcohol.sample(min(500, len(white_alcohol)), random_state=42))
    st.markdown(f"**Normality (Shapiro-Wilk):** Red p={p_red:.4f} | White p={p_white:.4f} — *Large samples rely on CLT; normality satisfied.*")

    lev_stat, lev_p = stats.levene(red_alcohol, white_alcohol)
    equal_var = lev_p > 0.05
    st.markdown(f"**Levene's Test:** stat={lev_stat:.4f}, p={lev_p:.4f} → {'Equal variance assumed' if equal_var else 'Unequal variance — using Welch t-test'}")

    t_stat, p_value = stats.ttest_ind(red_alcohol, white_alcohol, equal_var=equal_var)
    df_val = len(red_alcohol) + len(white_alcohol) - 2
    pooled_std = np.sqrt((red_alcohol.std()**2 + white_alcohol.std()**2) / 2)
    cohens_d   = abs(red_alcohol.mean() - white_alcohol.mean()) / pooled_std

    st.markdown("---")
    st.subheader("T-Test Results")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("T-Statistic",        f"{t_stat:.4f}")
    col2.metric("Degrees of Freedom", f"{df_val:,}")
    col3.metric("P-Value",            f"{p_value:.6f}")
    col4.metric("Cohen's d",          f"{cohens_d:.4f}")

    st.markdown("---")
    st.subheader("Interpretation")
    if p_value < alpha:
        st.success(f"✅ **Reject H₀** — p-value ({p_value:.6f}) < α ({alpha})")
        st.markdown(f"""
        There is a **statistically significant difference** in mean alcohol content.
        Red wine averages **{red_alcohol.mean():.2f}%** vs white wine **{white_alcohol.mean():.2f}%**.
        Effect size (Cohen's d = {cohens_d:.4f}) indicates a **{'small' if cohens_d < 0.2 else 'medium' if cohens_d < 0.5 else 'large'}** practical difference.
        """)
    else:
        st.warning(f"⚠️ **Fail to reject H₀** — p-value ({p_value:.6f}) ≥ α ({alpha})")

    st.markdown("---")
    st.subheader("Alcohol Distribution: Red vs White")
    fig, ax = light_fig(10, 4)
    ax.hist(red_alcohol,   bins=30, alpha=0.65, color=RED,  label=f"Red  (μ={red_alcohol.mean():.2f}%)", edgecolor="#ffffff")
    ax.hist(white_alcohol, bins=30, alpha=0.65, color=BLUE, label=f"White (μ={white_alcohol.mean():.2f}%)", edgecolor="#ffffff")
    ax.axvline(red_alcohol.mean(),   color=RED,  linestyle="--", linewidth=1.5)
    ax.axvline(white_alcohol.mean(), color=BLUE, linestyle="--", linewidth=1.5)
    ax.set_xlabel("Alcohol (%)")
    ax.set_ylabel("Count")
    ax.set_title("Alcohol Content Distribution by Wine Type")
    ax.legend(facecolor="#ffffff", edgecolor="#e8ddd8", labelcolor="#555555")
    plt.tight_layout()
    st.pyplot(fig); plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Predictive Modeling
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Predictive Modeling":
    st.markdown('<div class="section-tag">Momin</div>', unsafe_allow_html=True)
    st.title("Predictive Modeling")
    st.markdown('<p class="page-subtitle">Random Forest Classifier — predicting wine quality from chemical properties</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.subheader("Model Configuration")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        - **Algorithm:** Random Forest Classifier
        - **Estimators:** 100 decision trees
        - **Max Depth:** 10
        - **Train / Test Split:** 80% / 20% (stratified)
        """)
    with col2:
        st.markdown("""
        - **Target Variable:** `good_quality`
        - **Class 1:** Quality ≥ 7 (Good)
        - **Class 0:** Quality < 7 (Not Good)
        - **Class Weight:** Balanced (handles ~78% imbalance)
        """)

    st.markdown("---")
    accuracy = accuracy_score(y_test, y_pred)
    f1       = f1_score(y_test, y_pred)

    st.subheader("Model Performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy",  f"{accuracy*100:.2f}%")
    col2.metric("F1 Score",  f"{f1:.4f}")
    col3.metric("Test Size", f"{len(y_test):,} wines")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#fdf8f5")
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Not Good", "Good"])
        disp.plot(ax=ax, colorbar=False, cmap="Reds")
        ax.set_title("Confusion Matrix", color=WINE, fontsize=11)
        ax.tick_params(colors="#777777")
        ax.xaxis.label.set_color("#777777")
        ax.yaxis.label.set_color("#777777")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader("Feature Importances")
        importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True)
        fig, ax = light_fig(6, 4.5)
        colors = [WINE if i == len(importances)-1 else "#e8c4c4" for i in range(len(importances))]
        ax.barh(importances.index, importances.values, color=colors, edgecolor="#ffffff", height=0.6)
        ax.set_xlabel("Importance Score")
        ax.set_title("Feature Importances — Random Forest")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, target_names=["Not Good", "Good"], output_dict=True)
    st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

    st.markdown("---")
    st.subheader("Key Findings")
    col1, col2 = st.columns(2)
    with col1:
        st.info("🏆 **Alcohol** is the single most important predictor of wine quality.")
        st.info("📉 **Volatile acidity** and **density** are strong negative predictors.")
    with col2:
        st.info("⚗️ Chemical properties alone are sufficient to classify quality accurately.")
        st.info("📊 Balanced class weight significantly improved recall for good wines.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Quality Predictor
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Quality Predictor":
    st.markdown('<div class="section-tag">Interactive Tool</div>', unsafe_allow_html=True)
    st.title("Wine Quality Predictor")
    st.markdown('<p class="page-subtitle">Adjust the chemical properties below to predict whether a wine is good quality</p>', unsafe_allow_html=True)
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Chemistry I")
        wine_type      = st.selectbox("Wine Type", ["red", "white"])
        fixed_acidity  = st.slider("Fixed Acidity",    3.0, 16.0, 7.0, 0.1)
        volatile_acid  = st.slider("Volatile Acidity", 0.1, 1.6,  0.3, 0.01)
        citric_acid    = st.slider("Citric Acid",       0.0, 1.0,  0.3, 0.01)

    with col2:
        st.markdown("### Chemistry II")
        residual_sugar = st.slider("Residual Sugar",   0.5, 66.0, 5.0, 0.5)
        chlorides      = st.slider("Chlorides",        0.01, 0.6,  0.05, 0.001)
        free_so2       = st.slider("Free SO₂",         1.0, 100.0, 30.0, 1.0)
        total_so2      = st.slider("Total SO₂",        6.0, 300.0, 100.0, 1.0)

    with col3:
        st.markdown("### Chemistry III")
        density        = st.slider("Density",          0.990, 1.005, 0.996, 0.0001)
        ph             = st.slider("pH",               2.7, 4.0,   3.2,  0.01)
        sulphates      = st.slider("Sulphates",        0.2, 2.0,   0.5,  0.01)
        alcohol        = st.slider("Alcohol (%)",      8.0, 15.0,  10.5, 0.1)

    wine_type_enc = 0 if wine_type == "red" else 1
    input_data = pd.DataFrame([[
        fixed_acidity, volatile_acid, citric_acid, residual_sugar,
        chlorides, free_so2, total_so2, density, ph,
        sulphates, alcohol, wine_type_enc
    ]], columns=FEATURE_COLS)

    st.markdown("---")

    if st.button("Run Prediction", use_container_width=True):
        prediction  = rf.predict(input_data)[0]
        probability = rf.predict_proba(input_data)[0]

        st.markdown("<br>", unsafe_allow_html=True)
        if prediction == 1:
            st.success("✅  Good Quality Wine — quality score predicted ≥ 7")
        else:
            st.error("❌  Not Good Quality — quality score predicted < 7")

        col1, col2 = st.columns(2)
        col1.metric("Probability: Not Good", f"{probability[0]*100:.1f}%")
        col2.metric("Probability: Good",     f"{probability[1]*100:.1f}%")

        st.progress(float(probability[1]))
        st.caption(f"Model confidence this wine is good quality: {probability[1]*100:.1f}%")