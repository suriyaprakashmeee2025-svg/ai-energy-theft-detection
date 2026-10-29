"""Train, evaluate, and save the anomaly-detection model."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    classification_report, accuracy_score
)
from sklearn.model_selection import train_test_split
from generate_data import make_dataset

DATA_PATH = Path("data/meter_readings.csv")
OUT = Path("outputs")
REPORTS = Path("reports")
OUT.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

def add_features(df):
    """Create numeric features using only meter readings and context."""
    result = df.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce")
    result["hour"] = result["timestamp"].dt.hour
    result["day_of_week"] = result["timestamp"].dt.dayofweek
    result["month"] = result["timestamp"].dt.month
    result["is_weekend"] = (result["day_of_week"] >= 5).astype(int)
    result["rolling_mean_24h"] = (
        result.sort_values(["meter_id", "timestamp"])
        .groupby("meter_id")["consumption_kwh"]
        .transform(lambda s: s.rolling(24, min_periods=1).mean())
    )
    result["change_from_24h_mean"] = (
        result["consumption_kwh"] - result["rolling_mean_24h"]
    )
    result["consumption_per_meter_median_ratio"] = (
        result["consumption_kwh"] /
        result.groupby("meter_id")["consumption_kwh"].transform("median").clip(lower=0.01)
    )
    return result

def main():
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:
        df = make_dataset()
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_PATH, index=False)

    # Basic preprocessing.
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["consumption_kwh"] = pd.to_numeric(df["consumption_kwh"], errors="coerce")
    df = df.dropna(subset=["timestamp", "meter_id", "consumption_kwh"]).copy()
    df = df[df["consumption_kwh"] >= 0].copy()
    df = df.sort_values(["timestamp", "meter_id"]).reset_index(drop=True)

    # Split by time to avoid mixing future readings into the training set.
    cutoff = df["timestamp"].quantile(0.80)
    train_raw = df[df["timestamp"] <= cutoff].copy()
    test_raw = df[df["timestamp"] > cutoff].copy()

    train = add_features(train_raw)
    test = add_features(test_raw)

    features = [
        "consumption_kwh", "temperature_c", "hour", "day_of_week",
        "month", "is_weekend", "rolling_mean_24h",
        "change_from_24h_mean", "consumption_per_meter_median_ratio"
    ]
    # Fill any remaining missing numeric values using training medians.
    medians = train[features].median(numeric_only=True)
    X_train = train[features].replace([np.inf, -np.inf], np.nan).fillna(medians)
    X_test = test[features].replace([np.inf, -np.inf], np.nan).fillna(medians)
    y_test = test["is_simulated_anomaly"].astype(int)

    model = IsolationForest(
        n_estimators=200,
        contamination=0.02,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train)

    # sklearn decision_function: lower values are more anomalous.
    test["anomaly_score"] = -model.decision_function(X_test)
    test["flagged_anomaly"] = (model.predict(X_test) == -1).astype(int)

    y_pred = test["flagged_anomaly"]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "test_rows": int(len(test)),
        "known_simulated_anomalies_in_test": int(y_test.sum()),
        "flagged_rows": int(y_pred.sum()),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "true_positive": int(tp), "false_positive": int(fp),
        "true_negative": int(tn), "false_negative": int(fn),
        "note": "Metrics are measured against injected synthetic labels, not real theft cases."
    }

    # Save outputs.
    test.to_csv(OUT / "flagged_readings.csv", index=False)
    with open(OUT / "evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(OUT / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(features, f, indent=2)
    joblib.dump({"model": model, "feature_medians": medians.to_dict()}, OUT / "model.joblib")

    # Plot 1: consumption trend for a sample meter.
    sample_meter = df["meter_id"].iloc[0]
    sample = df[df["meter_id"] == sample_meter].sort_values("timestamp")
    plt.figure(figsize=(11, 4))
    plt.plot(sample["timestamp"], sample["consumption_kwh"], linewidth=0.8)
    flagged_idx = test[(test["meter_id"] == sample_meter) & (test["flagged_anomaly"] == 1)]
    if not flagged_idx.empty:
        plt.scatter(flagged_idx["timestamp"], flagged_idx["consumption_kwh"], s=18, label="Flagged")
        plt.legend()
    plt.title(f"Hourly consumption — {sample_meter}")
    plt.xlabel("Time"); plt.ylabel("Consumption (kWh)")
    plt.tight_layout(); plt.savefig(REPORTS / "consumption_trend.png", dpi=150); plt.close()

    plt.figure(figsize=(8, 4))
    plt.hist(test.loc[y_test == 0, "anomaly_score"], bins=40, alpha=0.7, label="Normal-labelled")
    plt.hist(test.loc[y_test == 1, "anomaly_score"], bins=40, alpha=0.7, label="Injected anomaly")
    plt.xlabel("Anomaly score (higher = more unusual)")
    plt.ylabel("Number of readings"); plt.title("Anomaly score distribution")
    plt.legend(); plt.tight_layout(); plt.savefig(REPORTS / "anomaly_scores.png", dpi=150); plt.close()

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion matrix")
    plt.xticks([0, 1], ["Predicted normal", "Predicted flagged"], rotation=15)
    plt.yticks([0, 1], ["Label normal", "Injected anomaly"])
    for (i, j), value in np.ndenumerate(cm):
        plt.text(j, i, str(value), ha="center", va="center")
    plt.tight_layout(); plt.savefig(REPORTS / "confusion_matrix.png", dpi=150); plt.close()

    print("\nMODEL EVALUATION (synthetic test data)")
    print(json.dumps(metrics, indent=2))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("\nSaved model and results in outputs/ and reports/")

if __name__ == "__main__":
    main()
