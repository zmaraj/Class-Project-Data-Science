# ml_model.py - Machine Learning Component
# Wine Quality Prediction using Random Forest Classifier

# ── imports ──────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.preprocessing import LabelEncoder
import joblib

# ── 1. Load cleaned data (produced by Zara's eda.py) ─────────────────────────
print("=" * 55)
print("  Wine Quality - Random Forest Classifier")
print("=" * 55)

df = pd.read_csv("data/wine_cleaned.csv")
print(f"\nDataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ── 2. Feature engineering ────────────────────────────────────────────────────
# Encode wine_type: red = 0, white = 1
le = LabelEncoder()
df["wine_type_encoded"] = le.fit_transform(df["wine_type"])

# Feature columns (all chemical properties + wine type)
FEATURE_COLS = [
    "fixed acidity", "volatile acidity", "citric acid",
    "residual sugar", "chlorides", "free sulfur dioxide",
    "total sulfur dioxide", "density", "pH", "sulphates",
    "alcohol", "wine_type_encoded"
]

TARGET_COL = "good_quality"   # 1 = quality >= 7, 0 = quality < 7

X = df[FEATURE_COLS]
y = df[TARGET_COL]

print(f"\nFeatures: {len(FEATURE_COLS)}")
print(f"Target  : good_quality  (1=good, 0=not good)")
print(f"\nClass distribution:")
print(f"  Good quality (1): {y.sum()} ({y.mean()*100:.1f}%)")
print(f"  Not good     (0): {(y==0).sum()} ({(y==0).mean()*100:.1f}%)")

# ── 3. Train / test split (80 / 20) ──────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set : {len(X_train)} rows")
print(f"Test  set : {len(X_test)} rows")

# ── 4. Train Random Forest ────────────────────────────────────────────────────
print("\nTraining Random Forest Classifier ...")

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight="balanced"   # handles class imbalance (~78% not good)
)
rf.fit(X_train, y_train)
print("Training complete.")

# ── 5. Evaluate ───────────────────────────────────────────────────────────────
y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1       = f1_score(y_test, y_pred)

print("\n" + "=" * 55)
print("  Model Performance")
print("=" * 55)
print(f"  Accuracy : {accuracy*100:.2f}%")
print(f"  F1 Score : {f1:.4f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Good", "Good"]))

# ── 6. Feature importance ─────────────────────────────────────────────────────
importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS)
importances = importances.sort_values(ascending=False)

print("Feature Importances (top 5):")
print(importances.head(5).round(4).to_string())

# ── 7. Save figures ───────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")

# Fig 7 - Confusion Matrix
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Not Good", "Good"])
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Fig 7 - Confusion Matrix\nRandom Forest Classifier", fontweight="bold")
plt.tight_layout()
plt.savefig("fig7_confusion_matrix.png", dpi=150)
plt.show()
print("Saved: fig7_confusion_matrix.png")

# Fig 8 - Feature Importance Bar Chart
fig, ax = plt.subplots(figsize=(9, 5))
colors = ["#c0392b" if i == 0 else "#2980b9" for i in range(len(importances))]
ax.barh(importances.index[::-1], importances.values[::-1], color=colors[::-1], edgecolor="white")
ax.set_title("Fig 8 - Feature Importances\nRandom Forest Classifier", fontweight="bold")
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig("fig8_feature_importance.png", dpi=150)
plt.show()
print("Saved: fig8_feature_importance.png")

# ── 8. Save trained model ─────────────────────────────────────────────────────
joblib.dump(rf, "data/rf_model.pkl")
joblib.dump(le, "data/label_encoder.pkl")
print("\nModel saved to: data/rf_model.pkl")
print("Label encoder saved to: data/label_encoder.pkl")

print("\nML component complete!")