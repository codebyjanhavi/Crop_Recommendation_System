from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np


# ==========================================
# 1. Create Flask application
# ==========================================

app = Flask(__name__)


# ==========================================
# 2. Load trained model and label encoder
# ==========================================

model = joblib.load(
    "models/random_forest_model.pkl"
)

label_encoder = joblib.load(
    "models/label_encoder.pkl"
)

print("Model loaded successfully!")
print("Label encoder loaded successfully!")


# ==========================================
# 3. Training dataset ranges
# ==========================================
# These ranges are based on the values present
# in the Crop Recommendation dataset.

FEATURE_RANGES = {

    "N": (0, 140),

    "P": (5, 145),

    "K": (5, 205),

    "temperature": (8.82, 43.68),

    "humidity": (14.25, 99.98),

    "ph": (3.50, 9.94),

    "rainfall": (20.21, 298.56)
}


# ==========================================
# 4. Home route
# ==========================================

@app.route("/")
def home():

    return render_template("index.html")


# ==========================================
# 5. Prediction route
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ------------------------------------------
        # Get JSON data from frontend
        # ------------------------------------------

        data = request.get_json()

        if not data:

            return jsonify({
                "error": "No input data received."
            }), 400


        # ------------------------------------------
        # Required input fields
        # ------------------------------------------

        required_fields = [
            "N",
            "P",
            "K",
            "temperature",
            "humidity",
            "ph",
            "rainfall"
        ]


        # ------------------------------------------
        # Check whether all fields are present
        # ------------------------------------------

        for field in required_fields:

            if field not in data:

                return jsonify({
                    "error":
                    f"Missing input field: {field}"
                }), 400


        # ------------------------------------------
        # Convert input values to float
        # ------------------------------------------

        values = {}

        for field in required_fields:

            try:

                values[field] = float(data[field])

            except (ValueError, TypeError):

                return jsonify({
                    "error":
                    f"{field} must be a valid number."
                }), 400


        # ------------------------------------------
        # Check training-data ranges
        # ------------------------------------------

        for field in required_fields:

            value = values[field]

            minimum, maximum = FEATURE_RANGES[field]


            if value < minimum or value > maximum:

                return jsonify({

                    "error":
                    f"{field} must be between "
                    f"{minimum} and {maximum} "
                    f"based on the training data."

                }), 400


        # ------------------------------------------
        # Arrange input in training order
        # ------------------------------------------

        input_data = np.array([
            [
                values["N"],
                values["P"],
                values["K"],
                values["temperature"],
                values["humidity"],
                values["ph"],
                values["rainfall"]
            ]
        ])


        # ------------------------------------------
        # Make prediction
        # ------------------------------------------

        prediction = model.predict(input_data)


        # ------------------------------------------
        # Convert encoded prediction to crop name
        # ------------------------------------------

        crop_name = label_encoder.inverse_transform(
            prediction
        )[0]


        # ------------------------------------------
        # Calculate model confidence
        # ------------------------------------------

        probabilities = model.predict_proba(
            input_data
        )

        confidence = (
            np.max(probabilities) * 100
        )


        # ------------------------------------------
        # Send result to frontend
        # ------------------------------------------

        return jsonify({

            "crop": crop_name,

            "confidence": round(
                confidence,
                2
            )

        })


    # ==========================================
    # Unexpected error
    # ==========================================

    except Exception as e:

        return jsonify({

            "error":
            f"An unexpected error occurred: {str(e)}"

        }), 500


# ==========================================
# 6. Run Flask application
# ==========================================

if __name__ == "__main__":

    app.run(
        debug=True
    )