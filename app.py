import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "car_price_dataset.csv"
MODEL_PATH = BASE_DIR / "car_price_model.joblib"
METRICS_PATH = BASE_DIR / "metrics.json"

st.set_page_config(page_title="Car Price Predictor", page_icon="🚘", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metrics():
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

data = load_data()
model = load_model()
metrics = load_metrics()
bm = metrics["best_metrics"]

st.title("🚘 Car Price Prediction Studio")
st.caption("A Machine Learning dashboard built with Python, Pandas, Scikit-learn and Streamlit.")
st.caption("Made by: Joy shill.")
st.info("This model learns from the supplied CSV dataset. Its output is an estimate in the dataset's price units, not a guaranteed current market quote.")

tab_predict, tab_insights, tab_model, tab_data = st.tabs(
    ["💰 Predict Price", "📊 Data Insights", "🧠 Model Performance", "🗂️ Dataset"]
)

with tab_predict:
    st.subheader("Enter vehicle details")
    st.write("Choose the vehicle features below, then click **Predict price**.")
    left, right = st.columns(2)
    with left:
        brand = st.selectbox("Brand", metrics["brands"])
        year = st.number_input("Model year", min_value=metrics["model_year_min"], max_value=metrics["model_year_max"], value=min(2020, metrics["model_year_max"]))
        engine_size = st.number_input("Engine size", min_value=float(metrics["engine_min"]), max_value=float(metrics["engine_max"]), value=float(round((metrics["engine_min"] + metrics["engine_max"]) / 2, 1)), step=0.1)
        fuel_type = st.selectbox("Fuel type", metrics["fuel_types"])
        transmission = st.selectbox("Transmission", metrics["transmissions"])
    with right:
        mileage = st.number_input("Mileage", min_value=metrics["mileage_min"], max_value=metrics["mileage_max"], value=int(data["Mileage"].median()), step=1000)
        doors = st.selectbox("Doors", metrics["doors_options"])
        owner_count = st.number_input("Owner count", min_value=metrics["owners_min"], max_value=metrics["owners_max"], value=int(data["Owner_Count"].median()), step=1)
        horsepower = st.number_input("Horsepower", min_value=metrics["horsepower_min"], max_value=metrics["horsepower_max"], value=int(data["Horsepower"].median()), step=1)

    if st.button("✨ Predict Car Price", type="primary", use_container_width=True):
        input_row = pd.DataFrame([{
            "Brand": brand,
            "Model_Year": int(year),
            "Engine_Size": float(engine_size),
            "Fuel_Type": fuel_type,
            "Transmission": transmission,
            "Mileage": int(mileage),
            "Doors": int(doors),
            "Owner_Count": int(owner_count),
            "Horsepower": int(horsepower)
        }])
        predicted = float(model.predict(input_row)[0])
        st.success("Prediction complete")
        st.metric("Estimated car price", f"{predicted:,.2f}")
        

with tab_insights:
    st.subheader("Explore the dataset")
    c1, c2, c3 = st.columns(3)
    c1.metric("Cars in dataset", f"{len(data):,}")
    c2.metric("Median listed price", f"{data['Price'].median():,.2f}")
    c3.metric("Missing values", f"{int(data.isna().sum().sum())}")
    st.write("### Price distribution")
    st.image(str(BASE_DIR / "price_distribution.png"), use_container_width=True)
    st.write("### Median price by brand")
    st.image(str(BASE_DIR / "median_price_by_brand.png"), use_container_width=True)

with tab_model:
    st.subheader("Held-out test performance")
    a, b, c = st.columns(3)
    a.metric("Best model", metrics["best_model"])
    b.metric("MAE", f"{bm['MAE']:,.2f}")
    c.metric("R²", f"{bm['R2']:.3f}")
    st.metric("RMSE", f"{bm['RMSE']:,.2f}")
    st.caption("MAE and RMSE are measured in the dataset's price units. R² is a goodness-of-fit score; results are specific to this dataset and split.")
    st.write("### Model comparison")
    comparison = pd.read_csv(BASE_DIR / "model_comparison.csv")
    st.dataframe(comparison, use_container_width=True, hide_index=True)
    st.write("### Actual vs predicted prices")
    st.image(str(BASE_DIR / "actual_vs_predicted.png"), use_container_width=True)

with tab_data:
    st.subheader("Dataset preview")
    st.dataframe(data.head(100), use_container_width=True, hide_index=True)
    st.download_button(
        "Download source CSV",
        data=DATA_PATH.read_bytes(),
        file_name="car_price_dataset.csv",
        mime="text/csv"
    )
    st.write("### Column summary")
    summary = pd.DataFrame({
        "column": data.columns,
        "data_type": [str(data[c].dtype) for c in data.columns],
        "missing_values": [int(data[c].isna().sum()) for c in data.columns],
        "unique_values": [int(data[c].nunique(dropna=False)) for c in data.columns]
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

st.divider()

