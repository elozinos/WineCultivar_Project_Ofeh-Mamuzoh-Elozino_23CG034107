import os
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# -------------------------
# Load model package
# -------------------------
model_package = joblib.load("model/wine_cultivar_model.pkl")
model = model_package["model"]
scaler = model_package["scaler"]
selected_features = model_package["features"]
feature_bounds = model_package["feature_bounds"]

# -------------------------
# Input validation
# -------------------------
def validate_inputs(user_inputs):
    for feature, value in user_inputs.items():
        min_val = feature_bounds[feature]["min"]
        max_val = feature_bounds[feature]["max"]
        if value < min_val or value > max_val:
            raise ValueError(
                f"{feature.replace('_',' ').title()} must be between {min_val:.2f} and {max_val:.2f}"
            )

# -------------------------
# Routes
# -------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            # Collect and validate inputs
            user_inputs = {feature: float(request.form[feature]) for feature in selected_features}
            validate_inputs(user_inputs)

            # Prepare input for model
            input_array = np.array([list(user_inputs.values())])
            input_scaled = scaler.transform(input_array)

            # Predict
            pred_class = model.predict(input_scaled)[0]
            prediction = f"Cultivar {pred_class + 1}"

        except Exception as e:
            error = str(e)

    return render_template("index.html", selected_features=selected_features, 
                           feature_bounds=feature_bounds, prediction=prediction, error=error)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use Render's assigned port or 5000 locally
    app.run(host="0.0.0.0", port=port)
