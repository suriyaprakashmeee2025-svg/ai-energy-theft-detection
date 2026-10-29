# AI-Based Energy Theft / Abnormal Consumption Pattern Detection

Beginner-friendly Second-Year EEE capstone project.

## What it does
- Uses simulated household electricity meter readings.
- Cleans and explores the readings.
- Builds time-based features.
- Trains an Isolation Forest anomaly-detection model.
- Produces anomaly scores and flags unusual intervals.
- Evaluates flagged intervals against simulated labels.
- Provides a Streamlit dashboard for demonstration.

**Responsible-use note:** A flag is a screening indicator, not proof of theft. Unusual usage may be caused by weather, visitors, holidays, meter faults, data errors, or a genuine change in household activity. Any real-world follow-up must be authorized and reviewed by a qualified person.

## 1. Install
Install Python 3.10+ if needed. Open a terminal in this project folder and run:

```bash
python -m venv .venv
```
Activate it:
- Windows: `.venv\Scripts\activate`
- macOS/Linux: `source .venv/bin/activate`

Then:
```bash
pip install -r requirements.txt
```

## 2. Generate sample data and train model
```bash
python train_model.py
```
This creates:
- `data/meter_readings.csv`
- `outputs/flagged_readings.csv`
- `outputs/evaluation_metrics.json`
- `outputs/model.joblib`
- `outputs/feature_columns.json`
- `reports/consumption_trend.png`
- `reports/anomaly_scores.png`
- `reports/confusion_matrix.png`

## 3. Run dashboard
```bash
streamlit run app.py
```
The dashboard runs locally in your browser.

## Dataset columns
- `timestamp`: interval timestamp
- `meter_id`: simulated meter identifier
- `consumption_kwh`: electricity used during the interval
- `temperature_c`: simulated outdoor temperature
- `day_type`: weekday or weekend
- `is_simulated_anomaly`: known injected anomaly label for evaluation only

The last column is available because this is a synthetic teaching dataset. It is NOT used as an input feature for model training.

## Model overview
Isolation Forest is an unsupervised anomaly detector. It learns patterns in the feature data and assigns an anomaly score. The project uses a separate labelled test set to report precision, recall, F1-score, and false-positive rate.

## Important limitations
- Simulated data is not evidence of actual theft.
- A model may flag legitimate changes in usage.
- Performance on synthetic data may not transfer to real utility data.
- Do not use model flags alone to accuse a customer, disconnect service, or impose a penalty.
- Real deployment requires consent/authorization, privacy safeguards, utility-approved data, validation, human review, and a documented appeal process.
