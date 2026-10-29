"""Create a reproducible synthetic electricity meter dataset for learning."""
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
OUT = Path("data/meter_readings.csv")

def make_dataset(n_days=120, n_meters=12, seed=SEED):
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2025-01-01", periods=n_days * 24, freq="h")
    rows = []
    for meter_num in range(1, n_meters + 1):
        meter_id = f"M{meter_num:03d}"
        household_scale = rng.uniform(0.55, 1.55)
        base = rng.uniform(0.12, 0.35)
        for ts in timestamps:
            hour = ts.hour
            weekend = ts.dayofweek >= 5
            # Typical morning and evening peaks.
            morning_peak = 0.30 * np.exp(-((hour - 7) / 2.0) ** 2)
            evening_peak = 0.65 * np.exp(-((hour - 19) / 3.0) ** 2)
            daytime = 0.12 if 9 <= hour <= 16 else 0.0
            weekend_factor = 1.12 if weekend else 1.0
            temperature = 27 + 5 * np.sin(2 * np.pi * (ts.dayofyear / 365)) + rng.normal(0, 1.5)
            cooling = max(0, temperature - 28) * 0.035
            expected = (base + morning_peak + evening_peak + daytime + cooling)
            expected *= household_scale * weekend_factor
            value = max(0.02, expected + rng.normal(0, 0.09 * household_scale))
            rows.append((ts, meter_id, value, temperature, weekend))

    df = pd.DataFrame(rows, columns=[
        "timestamp", "meter_id", "consumption_kwh", "temperature_c", "is_weekend"
    ])
    # Inject a small number of clearly unusual readings for educational evaluation.
    df["is_simulated_anomaly"] = 0
    rng = np.random.default_rng(seed + 1)
    candidates = df.index.to_numpy()
    chosen = rng.choice(candidates, size=max(30, int(0.012 * len(df))), replace=False)
    for idx in chosen:
        kind = rng.choice(["spike", "drop"])
        if kind == "spike":
            df.loc[idx, "consumption_kwh"] *= rng.uniform(3.5, 6.0)
        else:
            df.loc[idx, "consumption_kwh"] = max(0.01, df.loc[idx, "consumption_kwh"] * rng.uniform(0.03, 0.12))
        df.loc[idx, "is_simulated_anomaly"] = 1
    df["day_type"] = np.where(df["is_weekend"], "weekend", "weekday")
    df = df.drop(columns="is_weekend")
    return df.sort_values(["meter_id", "timestamp"]).reset_index(drop=True)

if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = make_dataset()
    data.to_csv(OUT, index=False)
    print(f"Saved {len(data):,} rows to {OUT}")
    print("Known injected anomalies:", int(data["is_simulated_anomaly"].sum()))
