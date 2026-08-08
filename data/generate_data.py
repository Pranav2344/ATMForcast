"""
generate_data.py
Generates synthetic historical ATM cash withdrawal data
for training the demand forecasting model.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

np.random.seed(42)

def generate_atm_data(start_date="2023-01-01", days=730, num_atms=5):
    records = []
    start = datetime.strptime(start_date, "%Y-%m-%d")

    # A few Indian festival dates that spike withdrawals (adjust yearly as needed)
    festival_dates = [
        "2023-01-14", "2023-03-08", "2023-08-15", "2023-10-24",
        "2023-11-12", "2023-12-25",
        "2024-01-14", "2024-03-25", "2024-08-15", "2024-10-31",
        "2024-11-01", "2024-12-25",
    ]
    festival_dates = set(pd.to_datetime(festival_dates))

    for atm_id in range(1, num_atms + 1):
        # Each ATM has a different baseline demand (location-based)
        base_demand = np.random.randint(150000, 400000)

        for i in range(days):
            date = start + timedelta(days=i)
            day_of_week = date.weekday()  # 0=Monday, 6=Sunday
            day_of_month = date.day

            demand = base_demand

            # Weekend effect (Sat/Sun higher footfall in India)
            if day_of_week in [5, 6]:
                demand *= np.random.uniform(1.15, 1.35)

            # Salary days effect (1st, and 28th-31st of month)
            if day_of_month <= 2 or day_of_month >= 28:
                demand *= np.random.uniform(1.4, 1.8)

            # Mid-month slight dip
            if 10 <= day_of_month <= 20:
                demand *= np.random.uniform(0.85, 0.95)

            # Festival spike
            if pd.Timestamp(date) in festival_dates:
                demand *= np.random.uniform(1.5, 2.0)

            # Random daily noise
            demand *= np.random.uniform(0.9, 1.1)

            # Occasional ATM downtime (zero demand days)
            if np.random.rand() < 0.01:
                demand = 0

            records.append({
                "date": date.strftime("%Y-%m-%d"),
                "atm_id": f"ATM_{atm_id:03d}",
                "day_of_week": day_of_week,
                "day_of_month": day_of_month,
                "cash_withdrawn": round(max(demand, 0), 2)
            })

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    df = generate_atm_data()

    # Ensure data folder exists
    os.makedirs(os.path.dirname(__file__), exist_ok=True)

    output_path = os.path.join(os.path.dirname(__file__), "atm_transactions.csv")
    df.to_csv(output_path, index=False)

    print(f"✅ Generated {len(df)} records for {df['atm_id'].nunique()} ATMs")
    print(f"✅ Saved to: {output_path}")
    print(df.head(10))