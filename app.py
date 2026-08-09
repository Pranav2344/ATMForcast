"""
app.py
Flask backend for ATM Cash Demand Forecasting dashboard.
"""

import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "atm_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "atm_transactions.csv")

app = Flask(__name__)

# ---- Load model + data once at startup ----
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
label_encoder = bundle["label_encoder"]
feature_cols = bundle["feature_cols"]

df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values(["atm_id", "date"]).reset_index(drop=True)

ATM_LIST = sorted(df["atm_id"].unique().tolist())


def build_features_for_date(atm_id, target_date, history_df):
    """Builds a single feature row for a given ATM and future date,
    using the most recent available history for lag/rolling features."""

    atm_hist = history_df[history_df["atm_id"] == atm_id].sort_values("date")

    day_of_week = target_date.weekday()
    day_of_month = target_date.day
    month = target_date.month
    is_weekend = int(day_of_week in [5, 6])
    is_salary_period = int(day_of_month <= 2 or day_of_month >= 28)
    is_month_start = int(day_of_month <= 5)
    is_month_end = int(day_of_month >= 25)

    atm_id_encoded = label_encoder.transform([atm_id])[0]

    lag_1 = atm_hist["cash_withdrawn"].iloc[-1]
    lag_7 = atm_hist["cash_withdrawn"].iloc[-7] if len(atm_hist) >= 7 else lag_1
    rolling_mean_7 = atm_hist["cash_withdrawn"].tail(7).mean()

    row = pd.DataFrame([{
        "atm_id_encoded": atm_id_encoded,
        "day_of_week": day_of_week,
        "day_of_month": day_of_month,
        "month": month,
        "is_weekend": is_weekend,
        "is_salary_period": is_salary_period,
        "is_month_start": is_month_start,
        "is_month_end": is_month_end,
        "lag_1": lag_1,
        "lag_7": lag_7,
        "rolling_mean_7": rolling_mean_7
    }])[feature_cols]

    return row


@app.route("/")
def index():
    return render_template("index.html", atm_list=ATM_LIST)


@app.route("/api/history/<atm_id>")
def get_history(atm_id):
    """Returns last 30 days of actual withdrawal history for an ATM."""
    atm_df = df[df["atm_id"] == atm_id].sort_values("date").tail(30)
    return jsonify({
        "dates": atm_df["date"].dt.strftime("%Y-%m-%d").tolist(),
        "values": atm_df["cash_withdrawn"].round(2).tolist()
    })


@app.route("/api/forecast/<atm_id>")
def forecast(atm_id):
    """Forecasts next 7 days of cash demand for the given ATM,
    feeding each day's prediction back in as history for the next."""
    if atm_id not in ATM_LIST:
        return jsonify({"error": "Invalid ATM ID"}), 400

    horizon = int(request.args.get("days", 7))
    working_df = df.copy()

    last_date = working_df[working_df["atm_id"] == atm_id]["date"].max()

    forecast_dates = []
    forecast_values = []

    for i in range(1, horizon + 1):
        target_date = last_date + timedelta(days=i)
        features = build_features_for_date(atm_id, target_date, working_df)
        pred = float(model.predict(features)[0])
        pred = max(pred, 0)

        forecast_dates.append(target_date.strftime("%Y-%m-%d"))
        forecast_values.append(round(pred, 2))

        # Append prediction to working history so next day's lag/rolling features use it
        new_row = pd.DataFrame([{
            "date": target_date,
            "atm_id": atm_id,
            "day_of_week": target_date.weekday(),
            "day_of_month": target_date.day,
            "cash_withdrawn": pred
        }])
        working_df = pd.concat([working_df, new_row], ignore_index=True)

    # Simple replenishment recommendation: forecasted demand + 15% safety buffer
    total_demand = sum(forecast_values)
    recommended_cash = round(total_demand * 1.15, 2)

    return jsonify({
        "atm_id": atm_id,
        "dates": forecast_dates,
        "values": forecast_values,
        "recommended_cash_load": recommended_cash
    })


@app.route("/api/atms")
def get_atms():
    return jsonify({"atms": ATM_LIST})

@app.route("/api/forecast-all")
def forecast_all():
    """Returns 7-day total forecast for every ATM, ranked by urgency."""
    horizon = int(request.args.get("days", 7))
    results = []

    for atm_id in ATM_LIST:
        working_df = df.copy()
        last_date = working_df[working_df["atm_id"] == atm_id]["date"].max()

        total_demand = 0
        for i in range(1, horizon + 1):
            target_date = last_date + timedelta(days=i)
            features = build_features_for_date(atm_id, target_date, working_df)
            pred = max(float(model.predict(features)[0]), 0)
            total_demand += pred

            new_row = pd.DataFrame([{
                "date": target_date,
                "atm_id": atm_id,
                "day_of_week": target_date.weekday(),
                "day_of_month": target_date.day,
                "cash_withdrawn": pred
            }])
            working_df = pd.concat([working_df, new_row], ignore_index=True)

        results.append({
            "atm_id": atm_id,
            "total_demand": round(total_demand, 2),
            "recommended_cash_load": round(total_demand * 1.15, 2)
        })

    # Sort by highest demand first = most urgent to refill
    results.sort(key=lambda x: x["total_demand"], reverse=True)

    return jsonify({"horizon_days": horizon, "atms": results})

@app.route("/api/upload-dataset", methods=["POST"])
def upload_dataset():
    """
    Local version: accepts a new CSV, merges it with the existing dataset,
    retrains the model, and overwrites the local files.
    """
    global df, model, label_encoder

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Use form field name 'file'."}), 400

    file = request.files["file"]
    try:
        new_df = pd.read_csv(file, parse_dates=["date"])
    except Exception as e:
        return jsonify({"error": f"Could not parse CSV: {str(e)}"}), 400

    required_cols = {"date", "atm_id", "day_of_week", "day_of_month", "cash_withdrawn"}
    if not required_cols.issubset(set(new_df.columns)):
        return jsonify({"error": f"CSV must contain columns: {sorted(required_cols)}"}), 400

    combined_df = pd.concat([df, new_df], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=["date", "atm_id"], keep="last")
    combined_df = combined_df.sort_values(["atm_id", "date"]).reset_index(drop=True)

    combined_df.to_csv(DATA_PATH, index=False)

    # Retrain on updated dataset
    import sys
    sys.path.append(os.path.join(BASE_DIR, "model"))
    from train_model import load_and_engineer_features
    from sklearn.preprocessing import LabelEncoder
    from sklearn.ensemble import HistGradientBoostingRegressor
    import joblib as jb

    engineered, le = load_and_engineer_features(DATA_PATH)
    feature_cols_local = bundle["feature_cols"]
    X = engineered[feature_cols_local]
    y = engineered["cash_withdrawn"]

    new_model = HistGradientBoostingRegressor(
        max_iter=400, max_depth=6, learning_rate=0.05,
        random_state=42
    )
    new_model.fit(X, y)

    jb.dump({"model": new_model, "label_encoder": le, "feature_cols": feature_cols_local}, MODEL_PATH)

    # Refresh in-memory objects used by other routes
    df = combined_df
    model = new_model
    label_encoder = le

    return jsonify({
        "message": "Dataset updated and model retrained successfully.",
        "total_rows": len(combined_df),
        "atms": sorted(combined_df["atm_id"].unique().tolist())
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
