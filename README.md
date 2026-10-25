# Car Price Prediction
### Machine Learning Project | Python • Scikit-learn • Streamlit

A beginner-friendly machine learning project that compares regression models and provides an interactive dashboard to estimate car prices from vehicle features.

---

## 1. Project Overview

This project uses a dataset of car specifications to train regression models that predict the `Price` of a car. The trained model is saved and used by a Streamlit dashboard, so you do not need to retrain it every time you open the app.

**Project type:** Regression  
**Target column:** `Price`  
**Dataset size:** 2,000 rows × 11 columns  
**Input identifier excluded:** `Car_ID`

> **Note:** The dataset does not specify a currency or price unit. Predictions should not be assumed to be INR or USD.

---

## 2. Project Structure

Keep these files together in the same project folder:

```text
Car-Price-Prediction/
├── app.py
├── train_model.py
├── car_price_dataset.csv
├── car_price_model.joblib
├── requirements.txt
├── metrics.json
├── model_comparison.csv
├── Project_Report.docx
├── Car_Price_Prediction_Presentation.pptx
└── README.md
```

The PNG chart files, if included, may also be used by the dashboard or report.

### What each file does

| File | Purpose |
|---|---|
| `car_price_dataset.csv` | Contains the car data used for training and evaluation. |
| `train_model.py` | Trains and compares regression models, then saves the best-performing pipeline. |
| `car_price_model.joblib` | Stores the trained model and its preprocessing steps. |
| `app.py` | Runs the interactive Streamlit prediction dashboard. |
| `requirements.txt` | Lists the Python packages needed by the project. |
| `metrics.json` | Stores evaluation metrics for the selected model. |
| `model_comparison.csv` | Contains the evaluation results for the compared models. |
| `Project_Report.docx` | Editable project report. |
| `Car_Price_Prediction_Presentation.pptx` | Editable project presentation. |

---

## 3. Dataset Information

- **Number of rows:** 2,000
- **Number of columns:** 11
- **Prediction target:** `Price`
- **Missing values in the supplied dataset:** 0
- **Identifier column:** `Car_ID` (excluded from model inputs)

The remaining columns describe vehicle characteristics, such as brand, model year, engine size, fuel type, transmission, mileage, doors, owner count, and horsepower.

---

## 4. Setup and Installation (Windows / VS Code)

### Step 1 — Extract the project

If you received a ZIP file, extract it first. Do not try to run Python files directly from inside the ZIP archive.

### Step 2 — Open the project folder

Open the extracted `Car-Price-Prediction` folder in VS Code. Then open **Terminal → New Terminal**.

### Step 3 — Create a virtual environment (recommended)

```bash
python -m venv .venv
```

Activate it in Windows Command Prompt:

```bat
.venv\Scripts\activate
```

If you use PowerShell, this activation command may be used instead:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 4 — Install the dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 5 — Start the dashboard

```bash
python -m streamlit run app.py
```

Streamlit will display a local address in the terminal. Open that address in your browser if it does not open automatically.

---

## 5. How to Use the Dashboard

1. Start the dashboard using the command above.
2. Enter or select the requested car features.
3. Click the prediction button shown in the app.
4. Review the estimated price displayed by the dashboard.

The exact input fields and button label depend on the implementation in `app.py`.

---

## 6. Model Training and Evaluation

The training workflow uses a fixed 80/20 train-test split with `random_state=42`.

- **80% of the data** is used for training.
- **20% of the data** is held out for testing.
- Categorical features are one-hot encoded.
- Numeric missing values are handled with a median imputer.
- Three regression algorithms are compared: Random Forest, Extra Trees, and Gradient Boosting.
- The model with the lowest test **MAE** is selected and saved as `car_price_model.joblib`.

### Evaluation metrics

| Metric | Meaning |
|---|---|
| **MAE** | Mean Absolute Error: the average absolute difference between predicted and actual prices. |
| **RMSE** | Root Mean Squared Error: gives more weight to larger prediction errors. |
| **R²** | Indicates how much variation in the target is explained by the model on the test data. |

Evaluation scores describe performance on the chosen dataset and split. They do not guarantee the same performance on real-world car listings.

---

## 7. Retrain the Model

You generally do **not** need to retrain the model every time you launch the dashboard. The saved `car_price_model.joblib` file is loaded by the app.

Retrain when you change the dataset, preprocessing, model algorithm, or model settings:

```bash
python train_model.py
```

This should recreate or update the saved model and evaluation files according to the code in `train_model.py`.

**Important:** Keep the training and dashboard environments compatible. If you see an error such as `ModuleNotFoundError: No module named '_loss'` while loading the saved model, check the installed scikit-learn version and retrain the model in the same environment used to run the dashboard.

---

## 8. Troubleshooting

### `ModuleNotFoundError` for a package

Activate the project virtual environment, then run:

```bash
python -m pip install -r requirements.txt
```

### The CSV file cannot be found

Make sure `car_price_dataset.csv` is in the location expected by `train_model.py`. If the script uses `Path(__file__).resolve().parent`, place the CSV in the same folder as that script.

### The saved model cannot be loaded

Run the training script in the same Python environment used for the dashboard:

```bash
python train_model.py
```

Then restart Streamlit.

### Streamlit command is not recognized

Try:

```bash
python -m streamlit run app.py
```

---

## 9. Limitations and Responsible Use

- This is an educational prototype, not a professional vehicle valuation service.
- The dataset's origin, currency, and real-world representativeness are not specified.
- Strong scores on one train-test split do not prove that the model will generalize to all car markets.
- Do not present predictions as guaranteed real-world prices without validation on reliable, representative data.

---

## 10. Suggested Future Improvements

- Evaluate the model with cross-validation.
- Tune hyperparameters and compare results on a consistent evaluation setup.
- Test with reliable real-world car listings from a clearly identified market.
- Add input validation and clearer units to the dashboard.
- Track model versions and evaluation results when retraining.

---

## Quick Commands

```bash
# Install dependencies
python -m pip install -r requirements.txt

# Run the dashboard
python -m streamlit run app.py

# Retrain and save the model
python train_model.py
```

---

**End of README**
