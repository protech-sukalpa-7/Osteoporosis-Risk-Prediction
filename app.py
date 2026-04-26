from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model, scaler, and encoders
model = joblib.load("model/osteoporosis_model.pkl")
scaler = joblib.load("model/scaler.pkl")
encoders = joblib.load("model/encoder.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = []

        # Collect form data
        for key in request.form:
            value = request.form[key]

            # Encode categorical data
            if key in encoders:
                value = encoders[key].transform([value])[0]
                features.append(value)
            else:
                features.append(float(value))

        # Convert to array
        final_features = np.array([features])

        # Scale features
        final_features = scaler.transform(final_features)

        # Prediction
        prediction = model.predict(final_features)[0]

        # Probability
        probability = model.predict_proba(final_features)[0][1]

        # Risk profiling
        if probability < 0.33:
            risk = "Low Risk"
        elif probability < 0.66:
            risk = "Medium Risk"
        else:
            risk = "High Risk"

        return render_template(
            "result.html",
            prediction=prediction,
            probability=round(probability * 100, 2),
            risk=risk
        )

    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    app.run(debug=True)