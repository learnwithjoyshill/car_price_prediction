"""
Car Price Prediction - Full Data Science Workflow
Operations:
1. Load and inspect CSV
2. Check missing values and duplicate rows
3. Clean data (remove exact duplicate rows; no arbitrary outlier removal)
4. Explore data and save charts
5. Split data into train/test sets
6. Preprocess numeric/categorical features
7. Train and compare 3 regression models
8. Save the best pipeline and metrics for the Streamlit UI

Run from the folder containing car_price_dataset.csv:
    python data_science_workflow.py
"""

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH =r"D:\python\car_price_dataset.csv"
TARGET = "Price"
ID_COLUMN = "Car_ID"
RANDOM_STATE = 42

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------
df = pd.read_csv(CSV_PATH)

print("\n========== DATASET OVERVIEW ==========")
print("First 5 rows:")
print(df.head())
print("\nShape (rows, columns):", df.shape)
print("\nColumn names:")
print(df.columns.tolist())
print("\nData types and non-null counts:")
df.info()
print("\nNumerical summary:")
print(df.describe())
print("\nMissing values per column:")
print(df.isna().sum())
print("\nDuplicate rows:", int(df.duplicated().sum()))

# ------------------------------------------------------------
# 2. BASIC DATA CLEANING
# ------------------------------------------------------------
# Remove exact duplicate records if any exist.
df = df.drop_duplicates().copy()

# Convert target to numeric; invalid target rows cannot train a regressor.
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df.dropna(subset=[TARGET]).copy()

# We do not automatically delete outliers: they should be investigated
# using domain knowledge before removal.
df.to_csv(BASE_DIR / "car_price_cleaned.csv", index=False)

print("\n========== AFTER BASIC CLEANING ==========")
print("Cleaned shape:", df.shape)
print("Missing values remaining:", int(df.isna().sum().sum()))

# ------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ------------------------------------------------------------
print("\n========== EDA ==========")
print("\nTarget Price summary:")
print(df[TARGET].describe())
for col in ["Brand", "Fuel_Type", "Transmission"]:
    if col in df.columns:
        print(f"\nValue counts for {col}:")
        print(df[col].value_counts(dropna=False))

# Price distribution
plt.figure(figsize=(8, 5))
df[TARGET].hist(bins=25, edgecolor="black")
plt.title("Distribution of Car Prices")
plt.xlabel("Price (dataset units)")
plt.ylabel("Number of cars")
plt.tight_layout()
plt.savefig(BASE_DIR / "price_distribution.png", dpi=160)
plt.close()

# Median price by brand
if "Brand" in df.columns:
    plt.figure(figsize=(9, 5))
    df.groupby("Brand")[TARGET].median().sort_values().plot(kind="bar")
    plt.title("Median Price by Brand")
    plt.xlabel("Brand")
    plt.ylabel("Median price (dataset units)")
    plt.tight_layout()
    plt.savefig(BASE_DIR / "median_price_by_brand.png", dpi=160)
    plt.close()

# Numeric correlation matrix (numeric features only)
numeric_for_corr = df.select_dtypes(include=np.number).drop(
    columns=[ID_COLUMN] if ID_COLUMN in df.columns else [], errors="ignore"
)
if numeric_for_corr.shape[1] > 1:
    corr = numeric_for_corr.corr()
    print("\nNumeric correlation matrix:")
    print(corr.round(3))
    plt.figure(figsize=(9, 7))
    plt.imshow(corr, aspect="auto")
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Numeric Feature Correlation")
    plt.tight_layout()
    plt.savefig(BASE_DIR / "correlation_matrix.png", dpi=160)
    plt.close()

# ------------------------------------------------------------
# 4. DEFINE FEATURES AND TARGET
# ------------------------------------------------------------
# Car_ID is an identifier, not a meaningful vehicle characteristic.
drop_columns = [TARGET]
if ID_COLUMN in df.columns:
    drop_columns.append(ID_COLUMN)

X = df.drop(columns=drop_columns)
y = df[TARGET]

categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_features = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

print("\nInput features:", X.columns.tolist())
print("Categorical features:", categorical_features)
print("Numeric features:", numeric_features)
print("Target:", TARGET)

# ------------------------------------------------------------
# 5. TRAIN/TEST SPLIT
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE
)
print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

# ------------------------------------------------------------
# 6. PREPROCESSING
# ------------------------------------------------------------
# Numeric missing values -> median.
# Categorical missing values -> most frequent category.
# Text categories -> one-hot encoding.
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# ------------------------------------------------------------
# 7. TRAIN AND COMPARE MODELS
# ------------------------------------------------------------
models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=300, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1
    ),
    "Extra Trees": ExtraTreesRegressor(
        n_estimators=300, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        random_state=RANDOM_STATE, n_estimators=150, learning_rate=0.05, max_depth=2
    ),
}

results = []
fitted_models = {}

for model_name, estimator in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator)
    ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append({
        "model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })
    fitted_models[model_name] = (pipeline, predictions)

results_df = pd.DataFrame(results).sort_values("MAE").reset_index(drop=True)
results_df.to_csv(BASE_DIR / "model_comparison.csv", index=False)

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

# Select the model with the lowest MAE on the held-out test set.
best_model_name = results_df.loc[0, "model"]
best_pipeline, best_predictions = fitted_models[best_model_name]
joblib.dump(best_pipeline, BASE_DIR / "car_price_model.joblib")

best_metrics = {
    "best_model": best_model_name,
    "MAE": float(mean_absolute_error(y_test, best_predictions)),
    "RMSE": float(np.sqrt(mean_squared_error(y_test, best_predictions))),
    "R2": float(r2_score(y_test, best_predictions)),
    "test_size": 0.20,
    "random_state": RANDOM_STATE,
    "target": TARGET,
    "features": X.columns.tolist(),
    "dataset_rows_after_cleaning": int(len(df)),
    "missing_values_after_cleaning": int(df.isna().sum().sum()),
    "price_currency": "Not specified in the supplied CSV"
}
(BASE_DIR / "metrics.json").write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")

# ------------------------------------------------------------
# 8. ACTUAL VS PREDICTED CHART
# ------------------------------------------------------------
plt.figure(figsize=(7, 6))
plt.scatter(y_test, best_predictions, alpha=0.65)
low = min(float(y_test.min()), float(np.min(best_predictions)))
high = max(float(y_test.max()), float(np.max(best_predictions)))
plt.plot([low, high], [low, high], linestyle="--")
plt.title(f"Actual vs Predicted Price - {best_model_name}")
plt.xlabel("Actual price")
plt.ylabel("Predicted price")
plt.tight_layout()
plt.savefig(BASE_DIR / "actual_vs_predicted.png", dpi=160)
plt.close()

print("\n========== BEST MODEL ==========")
print("Selected model:", best_model_name)
print(f"MAE:  {best_metrics['MAE']:.2f}")
print(f"RMSE: {best_metrics['RMSE']:.2f}")
print(f"R2:   {best_metrics['R2']:.4f}")
print("\nSaved files:")
print("- car_price_cleaned.csv")
print("- price_distribution.png")
print("- median_price_by_brand.png")
print("- correlation_matrix.png (if numeric features are available)")
print("- model_comparison.csv")
print("- car_price_model.joblib")
print("- metrics.json")
print("- actual_vs_predicted.png")