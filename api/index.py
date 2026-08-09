"""
api/index.py
Flask backend for ATM Cash Demand Forecasting — Vercel deployment.
Reads/writes dataset + model from Vercel Blob Storage instead of local disk.
"""

import os
import io
import joblib
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.ensemble import HistGradientBoostingRegressor
from flask import Flask, render_template, request, jsonify
import vercel_blob

app = Flask(
    __name__,
    template_folder="../templates",
    static_folder="../static"
)

DATASET_BLOB_NAME = "atm_transactions.csv"
MODEL_BLOB_NAME = "atm_model.pkl"

FEATURE_COLS = [
    "atm_id_encoded", "day_of_week", "day_of_month", "month",
    "is_weekend", "is_salary_period", "is_month_start", "is_month_end",
    "lag_1", "lag_7", "rolling_mean_7"
]

# In-memory cache (persists only within a warm function instance)
_cache = {"df": None, "model": None, "label_encoder": None}


def find_blob_url(pathname):
    """Looks up a blob's current URL by its pathname."""
    blobs = vercel_blob.list({}).get("blobs", [])
    for b in blobs:
        if b["pathname"] == pathname:
            return b["url"]
    return None


def load_dataset():
    if _cache["df"] is not None:
        return _cache["df"]

    url = find_blob_url(DATASET_BLOB_NAME)
    if url is None:
        raise FileNotFoundError("No dataset found in Blob storage yet. Upload one first.")

    df = pd.read_csv(url, parse_dates=["date"])
    df = df.sort_values(["atm_id", "date"]).reset_index(drop=True)
    _cache["df"] = df
    return df


def load_model():
    if _cache["model"] is not None:
        return _cache["model"], _cache["label_encoder"]

    url = find_blob_url(MODEL_BLOB_NAME)
    if url is None:
        raise FileNotFoundError("No trained model found in Blob storage yet.")

    import urllib.request
    with urllib.request.urlopen(url) as response:
        bundle = joblib.load(io.BytesIO(response.read()))

    _cache["model"] = bundle["model"]
    _cache["label_encoder"] = bundle["label_encoder"]
    return _cache["model"], _cache["label_encoder"]


def engineer_features(df):
    df = df.copy()
    df["month"] = df["date"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_salary_period"] = ((df["day_of_month"] <= 2) | (df["day_of_month"] >= 28)).astype(int)
    df["is_month_start"] = (df["day_of_month"] <= 5).astype(int)
    df["is_month_end"] = (df["day_of_month"] >= 25).astype(int)

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df["atm_id_encoded"] = le.fit_transform(df["atm_id"])

    df["lag_1"] = df.groupby("atm_id")["cash_withdrawn"].shift(1)
    df["lag_7"] = df.groupby("atm_id")["cash_withdrawn"].shift(7)
    df["rolling_mean_7"] = (
        df.groupby("atm_id")["cash_withdrawn"]
        .transform(lambda x: x.shift(1).rolling(window=7, min_periods=1).mean())
    )
    df = df.dropna(subset=["lag_1", "lag_7", "rolling_mean_7"]).reset_index(drop=True)
    return df, le


def retrain_and_save(df):
    """Retrains the model on the given dataframe and saves it to Blob storage."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    engineered, le = engineer_features(df)
    X = engineered[FEATURE_COLS]
    y = engineered["cash_withdrawn"]

    model = HistGradientBoostingRegressor(
        max_iter=400, max_depth=6, learning_rate=0.05,
        random_state=42
    )
    model.fit(X, y)

    buffer = io.BytesIO()
    joblib.dump({"model": model, "label_encoder": le, "feature_cols": FEATURE_COLS}, buffer)
    buffer.seek(0)

    vercel_blob.put(MODEL_BLOB_NAME, buffer.read(), {"addRandomSuffix": "false"})

    _cache["model"] = model
    _cache["label_encoder"] = le

    return model, le


def build_features_for_date(atm_id, target_date, history_df, label_encoder):
    atm_hist = history_df[history_df["atm_id"] == atm_id].sort_values("date")

    atm_id_encoded = label_encoder.transform([atm_id])[0]
    lag_1 = atm_hist["cash_withdrawn"].iloc[-1]
    lag_7 = atm_hist["cash_withdrawn"].iloc[-7] if len(atm_hist) >= 7 else lag_1
    rolling_mean_7 = atm_hist["cash_withdrawn"].tail(7).mean()

    row = pd.DataFrame([{
        "atm_id_encoded": atm_id_encoded,
        "day_of_week": target_date.weekday(),
        "day_of_month": target_date.day,
        "month": target_date.month,
        "is_weekend": int(target_date.weekday() in [5, 6]),
        "is_salary_period": int(target_date.day <= 2 or target_date.day >= 28),
        "is_month_start": int(target_date.day <= 5),
        "is_month_end": int(target_date.day >= 25),
        "lag_1": lag_1, "lag_7": lag_7, "rolling_mean_7": rolling_mean_7
    }])[FEATURE_COLS]
    return row


@app.route("/")
def index():
    try:
        df = load_dataset()
        atm_list = sorted(df["atm_id"].unique().tolist())
    except FileNotFoundError:
        atm_list = []
    return render_template("index.html", atm_list=atm_list)


@app.route("/api/atms")
def get_atms():
    df = load_dataset()
    return jsonify({"atms": sorted(df["atm_id"].unique().tolist())})


@app.route("/api/history/<atm_id>")
def get_history(atm_id):
    df = load_dataset()
    atm_df = df[df["atm_id"] == atm_id].sort_values("date").tail(30)
    return jsonify({
        "dates": atm_df["date"].dt.strftime("%Y-%m-%d").tolist(),
        "values": atm_df["cash_withdrawn"].round(2).tolist()
    })


@app.route("/api/forecast/<atm_id>")
def forecast(atm_id):
    df = load_dataset()
    model, label_encoder = load_model()

    horizon = int(request.args.get("days", 7))
    working_df = df.copy()
    last_date = working_df[working_df["atm_id"] == atm_id]["date"].max()

    forecast_dates, forecast_values = [], []
    for i in range(1, horizon + 1):
        target_date = last_date + timedelta(days=i)
        features = build_features_for_date(atm_id, target_date, working_df, label_encoder)
        pred = max(float(model.predict(features)[0]), 0)
        forecast_dates.append(target_date.strftime("%Y-%m-%d"))
        forecast_values.append(round(pred, 2))

        new_row = pd.DataFrame([{
            "date": target_date, "atm_id": atm_id,
            "day_of_week": target_date.weekday(), "day_of_month": target_date.day,
            "cash_withdrawn": pred
        }])
        working_df = pd.concat([working_df, new_row], ignore_index=True)

    total_demand = sum(forecast_values)
    return jsonify({
        "atm_id": atm_id, "dates": forecast_dates, "values": forecast_values,
        "recommended_cash_load": round(total_demand * 1.15, 2)
    })


@app.route("/api/forecast-all")
def forecast_all():
    df = load_dataset()
    model, label_encoder = load_model()
    horizon = int(request.args.get("days", 7))
    results = []

    for atm_id in sorted(df["atm_id"].unique().tolist()):
        working_df = df.copy()
        last_date = working_df[working_df["atm_id"] == atm_id]["date"].max()
        total_demand = 0
        for i in range(1, horizon + 1):
            target_date = last_date + timedelta(days=i)
            features = build_features_for_date(atm_id, target_date, working_df, label_encoder)
            pred = max(float(model.predict(features)[0]), 0)
            total_demand += pred
            new_row = pd.DataFrame([{
                "date": target_date, "atm_id": atm_id,
                "day_of_week": target_date.weekday(), "day_of_month": target_date.day,
                "cash_withdrawn": pred
            }])
            working_df = pd.concat([working_df, new_row], ignore_index=True)

        results.append({
            "atm_id": atm_id,
            "total_demand": round(total_demand, 2),
            "recommended_cash_load": round(total_demand * 1.15, 2)
        })

    results.sort(key=lambda x: x["total_demand"], reverse=True)
    return jsonify({"horizon_days": horizon, "atms": results})


@app.route("/api/upload-dataset", methods=["POST"])
def upload_dataset():
    """
    Accepts a new CSV (same columns: date, atm_id, day_of_week, day_of_month, cash_withdrawn),
    merges it with the existing dataset in Blob storage, retrains the model,
    and saves both back to Blob.
    """
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

    # Merge with existing dataset (if any), drop exact duplicate rows
    try:
        existing_df = load_dataset()
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    except FileNotFoundError:
        combined_df = new_df

    combined_df = combined_df.drop_duplicates(subset=["date", "atm_id"], keep="last")
    combined_df = combined_df.sort_values(["atm_id", "date"]).reset_index(drop=True)

    # Save merged dataset back to Blob
    csv_buffer = io.StringIO()
    combined_df.to_csv(csv_buffer, index=False)
    vercel_blob.put(DATASET_BLOB_NAME, csv_buffer.getvalue().encode(), {"addRandomSuffix": "false"})

    _cache["df"] = combined_df

    # Retrain model on the updated dataset
    retrain_and_save(combined_df)

    return jsonify({
        "message": "Dataset updated and model retrained successfully.",
        "total_rows": len(combined_df),
        "atms": sorted(combined_df["atm_id"].unique().tolist())
    })


# Vercel's Python runtime looks for a variable named 'app'
