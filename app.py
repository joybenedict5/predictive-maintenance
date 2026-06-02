from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

model    = joblib.load('pm_model.pkl')
scaler   = joblib.load('pm_scaler.pkl')
features = joblib.load('pm_features.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    missing = [f for f in features if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    values = np.array([[data[f] for f in features]])
    scaled = scaler.transform(values)

    prediction  = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0]

    return jsonify({
        "failure_predicted": bool(prediction),
        "failure_probability": round(float(probability[1]) * 100, 2),
        "status": "ALERT: Failure imminent!" if prediction == 1 else "OK: Engine healthy",
    }), 200

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "running"}), 200

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=False, host='0.0.0.0', port=port)