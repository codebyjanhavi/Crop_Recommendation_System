from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import json

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

# Load model comparison results

with open(
    "model_comparison.json",
    "r"
) as file:

    model_comparison = json.load(file)

print("Model loaded successfully!")
print("Label encoder loaded successfully!")


# ==========================================
# 3. Training dataset ranges
# ==========================================

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
# Feature importance
# ==========================================

FEATURE_NAMES = [
    "Nitrogen",
    "Phosphorus",
    "Potassium",
    "Temperature",
    "Humidity",
    "Soil pH",
    "Rainfall"
]


feature_importance = []

for name, importance in zip(
    FEATURE_NAMES,
    model.feature_importances_
):

    feature_importance.append({

        "feature": name,

        "importance": round(
            float(importance) * 100,
            2
        )

    })


feature_importance.sort(
    key=lambda x: x["importance"],
    reverse=True
)

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
                    "error": f"Missing input field: {field}"
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
                    "error": f"{field} must be a valid number."
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
        # Calculate prediction probabilities
        # ------------------------------------------

        probabilities = model.predict_proba(
            input_data
        )[0]


        # ------------------------------------------
        # Get indices of top 3 predictions
        # ------------------------------------------

        top_indices = np.argsort(
            probabilities
        )[::-1][:3]


        # ------------------------------------------
        # Convert encoded predictions to crop names
        # ------------------------------------------

        top_crops = label_encoder.inverse_transform(
            model.classes_[top_indices]
        )


        # ------------------------------------------
        # Create top 3 recommendations
        # ------------------------------------------

        recommendations = []

        for crop, probability in zip(
            top_crops,
            probabilities[top_indices]
        ):

            recommendations.append({

                "crop": crop,

                "confidence": round(
                    probability * 100,
                    2
                )

            })


        # ------------------------------------------
        # Best crop confidence
        # ------------------------------------------

        confidence = (
            probabilities[top_indices[0]] * 100
        )


        # ------------------------------------------
        # Send result to frontend
        # ------------------------------------------

        return jsonify({

            "crop": crop_name,

            "confidence": round(
                confidence,
                2
            ),

            "recommendations": recommendations

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
# Model comparison route
# ==========================================

@app.route("/model-comparison")
def model_comparison_page():

    return jsonify(model_comparison)


# ==========================================
# 6. Run Flask application
# ==========================================

if __name__ == "__main__":

    app.run(
        debug=True
    )