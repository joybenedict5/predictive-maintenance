# Predictive Maintenance ML System

A machine learning system that predicts industrial engine failures before they happen, helping factories avoid unexpected breakdowns.

## Features
- Trained on NASA turbofan sensor data (simulated)
- Random Forest classifier with 300 estimators and class balancing
- 100% Recall on failure cases — never misses a failure
- RUL (Remaining Useful Life) feature engineered from scratch
- SHAP-based feature importance — reduced inputs by 30%
- Flask REST API for real-time sensor predictions

## Tech Stack
Python · scikit-learn · Flask · pandas · NumPy · joblib

## Setup
```bash
pip install -r requirements.txt
python feature_engineering.py
python model_training.py
python shap_analysis.py
python app.py
```

## API Usage
```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"sensor_1": 550, "sensor_2": 641, "sensor_3": 1580, "sensor_4": 1400, "sensor_5": 14, "sensor_6": 21, "sensor_7": 554, "sensor_8": 2388, "sensor_9": 9065, "sensor_10": 1.3}'
```

## Results
| Metric | Score |
|--------|-------|
| Recall (Failure Detection) | 100% |
| Overall Accuracy | 91% |
| Feature Reduction (SHAP) | 30% |
| Noisy Sample Reduction | 22% |