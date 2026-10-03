import joblib
import numpy as np


# ==========================================
# 1. Load saved model and label encoder
# ==========================================

model = joblib.load("models/random_forest_model.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")


# ==========================================
# 2. Function to safely get numeric input
# ==========================================

def get_number(message, minimum=None, maximum=None):

    while True:

        try:
            value = float(input(message))

            # Check minimum value
            if minimum is not None and value < minimum:
                print(
                    f"Value must be at least {minimum}. "
                    "Please try again."
                )
                continue

            # Check maximum value
            if maximum is not None and value > maximum:
                print(
                    f"Value must be at most {maximum}. "
                    "Please try again."
                )
                continue

            return value

        except ValueError:
            print("Invalid input! Please enter a number.")


# ==========================================
# 3. Display title
# ==========================================

print("========================================")
print("       CROP RECOMMENDATION SYSTEM")
print("========================================")


# ==========================================
# 4. Take input from user
# ==========================================

N = get_number(
    "Enter Nitrogen (N): ",
    minimum=0
)

P = get_number(
    "Enter Phosphorus (P): ",
    minimum=0
)

K = get_number(
    "Enter Potassium (K): ",
    minimum=0
)

temperature = get_number(
    "Enter Temperature (°C): "
)

humidity = get_number(
    "Enter Humidity (%): ",
    minimum=0,
    maximum=100
)

ph = get_number(
    "Enter Soil pH: ",
    minimum=0,
    maximum=14
)

rainfall = get_number(
    "Enter Rainfall (mm): ",
    minimum=0
)


# ==========================================
# 5. Arrange input in correct order
# ==========================================

input_data = np.array([
    [
        N,
        P,
        K,
        temperature,
        humidity,
        ph,
        rainfall
    ]
])


# ==========================================
# 6. Make prediction
# ==========================================

prediction = model.predict(input_data)


# ==========================================
# 7. Calculate model confidence
# ==========================================

probabilities = model.predict_proba(input_data)

max_probability = np.max(probabilities)

confidence = max_probability * 100


# ==========================================
# 8. Convert prediction number to crop name
# ==========================================

crop_name = label_encoder.inverse_transform(prediction)


# ==========================================
# 9. Display result
# ==========================================

print("\n========================================")
print("        CROP RECOMMENDATION")
print("========================================")

print("Recommended Crop:", crop_name[0])
print(f"Model Confidence: {confidence:.2f}%")

print("========================================")