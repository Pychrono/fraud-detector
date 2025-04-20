from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)
model = joblib.load('models/fraud_model.pkl')
simple_model = joblib.load("models/simple_fraud_model.pkl")


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict-simple', methods=['POST'])
def predict_simple():
    try:
        data = request.get_json()
        features = [
            data['amount'],
            data['is_foreign'],
            data['is_high_risk_country'],
            data['used_chip']
        ]
        prediction = simple_model.predict([features])[0]
        result = 'fraud' if prediction == 1 else 'not fraud'
        return jsonify({'prediction': int(prediction), 'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    try:
        data = request.get_json()

        # Convert simplified input to fake V1–V28 values (basic logic just for demo)
        # You could later train a lightweight model using only these 4 features.
        v_features = [0.0] * 28  # Placeholder for V1–V28
        amount = data.get('amount', 0)
        features = v_features + [amount]

        prediction = model.predict([features])[0]
        result = 'fraud' if prediction == 1 else 'not fraud'
        return jsonify({'prediction': int(prediction), 'result': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict', methods=['POST'])
def predict_advanced():
    try:
        data = request.get_json()

        # Expecting all 28 V features + Amount
        input_features = [data[f"V{i}"] for i in range(1, 29)] + [data['Amount']]

        prediction = model.predict([input_features])[0]
        result = 'fraud' if prediction == 1 else 'not fraud'
        return jsonify({'prediction': int(prediction), 'result': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

import os

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
