"""Local Streamlit dashboard for the capstone project."""
from pathlib import Path
import json
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AI Energy Anomaly Monitor", layout="wide")
st.title("AI-Based Energy Theft / Abnormal Consumption Pattern Detection")
st.caption("Educational screening dashboard — a flag is not proof of theft.")

data_path = Path("outputs/flagged_readings.csv")
metrics_path = Path("outputs/evaluation_metrics.json")
if not data_path.exists() or not metrics_path.exists():
    st.warning("Model outputs are missing. Run `python train_model.py` first.")
    st.stop()

df = pd.read_csv(data_path, parse_dates=["timestamp"])
metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

with st.sidebar:
    st.header("Filters")
    meters = sorted(df["meter_id"].dropna().unique().tolist())
    selected_meters = st.multiselect("Meter IDs", meters, default=meters[:3] if len(meters) > 3 else meters)
    show_flagged_only = st.checkbox("Show flagged readings only", value=False)
    st.info("The labels in this demo are simulated and are used only to evaluate the teaching model.")

filtered = df[df["meter_id"].isin(selected_meters)].copy()
if show_flagged_only:
    filtered = filtered[filtered["flagged_anomaly"] == 1]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Test readings", f"{metrics['test_rows']:,}")
c2.metric("Flagged readings", f"{metrics['flagged_rows']:,}")
c3.metric("Precision", f"{metrics['precision']:.1%}")
c4.metric("False-positive rate", f"{metrics['false_positive_rate']:.1%}")

st.subheader("Consumption trend")
if not filtered.empty:
    st.line_chart(filtered.sort_values("timestamp").set_index("timestamp")["consumption_kwh"])
else:
    st.info("No readings match the selected filters.")

st.subheader("Flagged intervals")
flagged = filtered[filtered["flagged_anomaly"] == 1].sort_values("anomaly_score", ascending=False)
cols = [c for c in ["timestamp", "meter_id", "consumption_kwh", "anomaly_score",
                    "flagged_anomaly", "is_simulated_anomaly"] if c in flagged.columns]
st.dataframe(flagged[cols], use_container_width=True, hide_index=True)

st.subheader("Evaluation")
st.json(metrics)
st.markdown("""
### How to interpret the result
- **Anomaly score:** larger values mean the model considers the reading more unusual.
- **Flagged reading:** a candidate for review, not a theft verdict.
- **False positive:** a normal-labelled reading flagged by the model.
- **False negative:** a simulated anomaly the model missed.

### Responsible use
A real utility should verify meter condition, data quality, weather, household changes, and other legitimate explanations. Human review and authorized procedures are required before any action affecting a customer.
""")
