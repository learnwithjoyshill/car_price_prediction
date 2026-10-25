from pathlib import Path
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / "car_price_dataset.csv")

target = "Price"
X = df.drop(columns=[target, "Car_ID"], errors="ignore")
y = df[target]
categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
numeric = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical)
])

candidates = {
    "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1),
    "Extra Trees": ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42, n_estimators=150, learning_rate=0.05, max_depth=2)
}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rows, fitted = [], []
for name, estimator in candidates.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    rows.append({
        "model": name,
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        "R2": r2_score(y_test, pred)
    })
    fitted.append((name, pipe, pred))

results = pd.DataFrame(rows).sort_values("MAE")
best_name = results.iloc[0]["model"]
best_pipe = next(p for name, p, pred in fitted if name == best_name)
joblib.dump(best_pipe, BASE_DIR / "car_price_model.joblib")
results.to_csv(BASE_DIR / "model_comparison.csv", index=False)
print(results.to_string(index=False))
print(f"Saved best model: {best_name}")
