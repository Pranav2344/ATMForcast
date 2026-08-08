"""
train_model.py
Trains an XGBoost regression model to forecast daily ATM cash demand.
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "atm_transactions.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "atm_model.pkl")


def load_and_engineer_features(path):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values(["atm_id", "date"]).reset_index(drop=True)

    # Calendar features
    df["month"] = df["date"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_salary_period"] = ((df["day_of_month"] <= 2) | (df["day_of_month"] >= 28)).astype(int)
    df["is_month_start"] = (df["day_of_month"] <= 5).astype(int)
    df["is_month_end"] = (df["day_of_month"] >= 25).astype(int)

    # Encode atm_id
    le = LabelEncoder()
    df["atm_id_encoded"] = le.fit_transform(df["atm_id"])

    # Lag features (previous day, previous week) — grouped per ATM
    df["lag_1"] = df.groupby("atm_id")["cash_withdrawn"].shift(1)
    df["lag_7"] = df.groupby("atm_id")["cash_withdrawn"].shift(7)

    # Rolling average (7-day) — shifted so we don't leak today's value
    df["rolling_mean_7"] = (
        df.groupby("atm_id")["cash_withdrawn"]
        .transform(lambda x: x.shift(1).rolling(window=7, min_periods=1).mean())
    )

    # Drop rows where lag features are NaN (first week of each ATM)
    df = df.dropna(subset=["lag_1", "lag_7", "rolling_mean_7"]).reset_index(drop=True)

    return df, le


def train():
    print("📥 Loading data...")
    df, label_encoder = load_and_engineer_features(DATA_PATH)

    feature_cols = [
        "atm_id_encoded", "day_of_week", "day_of_month", "month",
        "is_weekend", "is_salary_period", "is_month_start", "is_month_end",
        "lag_1", "lag_7", "rolling_mean_7"
    ]
    target_col = "cash_withdrawn"

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False  # keep time order for realistic eval
    )

    print("🧠 Training XGBoost model...")
    model = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print("\n📊 Evaluation on test set:")
    print(f"MAE  : ₹{mae:,.2f}")
    print(f"RMSE : ₹{rmse:,.2f}")
    print(f"R²   : {r2:.4f}")

    # Save model + encoder + feature list together
    joblib.dump({
        "model": model,
        "label_encoder": label_encoder,
        "feature_cols": feature_cols
    }, MODEL_PATH)

    print(f"\n✅ Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train()