
import json
from datetime import date
from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# 1. LOAD TRAINED MODEL
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "guest_shield_model.joblib"

# Load once, outside the Lambda handler.
# Warm Lambda executions can reuse this object.
MODEL_BUNDLE = joblib.load(MODEL_PATH)

PREPROCESSOR = MODEL_BUNDLE["preprocessor"]
KMEANS = MODEL_BUNDLE["kmeans"]
MODELS = MODEL_BUNDLE["models"]
LABEL_ENCODER = MODEL_BUNDLE["label_encoder"]
FEATURE_NAMES = MODEL_BUNDLE["feature_names"]

CATEGORICAL_COLS = MODEL_BUNDLE["categorical_columns"]
NUMERICAL_COLS = MODEL_BUNDLE["numerical_columns"]


# ============================================================
# 2. VALIDATE AND PREPARE INPUT
# ============================================================

def prepare_reservation(reservation):
    """Validate a reservation and recreate training features."""

    if not isinstance(reservation, dict):
        raise ValueError("Reservation must be a JSON object.")

    # These are the original model input fields.
    required_fields = [
        "no_of_adults",
        "no_of_children",
        "no_of_weekend_nights",
        "no_of_week_nights",
        "type_of_meal_plan",
        "required_car_parking_space",
        "room_type_reserved",
        "lead_time",
        "arrival_year",
        "arrival_month",
        "arrival_date",
        "market_segment_type",
        "repeated_guest",
        "no_of_previous_cancellations",
        "no_of_previous_bookings_not_canceled",
        "avg_price_per_room",
        "no_of_special_requests",
    ]

    missing = [
        field
        for field in required_fields
        if field not in reservation
    ]

    if missing:
        raise ValueError(
            "Missing required fields: " + ", ".join(missing)
        )

    row = dict(reservation)

    # Check basic numerical constraints.
    nonnegative_fields = [
        "no_of_adults",
        "no_of_children",
        "no_of_weekend_nights",
        "no_of_week_nights",
        "lead_time",
        "no_of_previous_cancellations",
        "no_of_previous_bookings_not_canceled",
        "avg_price_per_room",
        "no_of_special_requests",
    ]

    for field in nonnegative_fields:
        value = row[field]

        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0 <= value < float("inf")
        ):
            raise ValueError(
                f"{field} must be a finite, non-negative number."
            )

    # Validate binary categorical fields.
    for field in [
        "required_car_parking_space",
        "repeated_guest",
    ]:
        if type(row[field]) is not int or row[field] not in (0, 1):
            raise ValueError(f"{field} must be 0 or 1.")

    # Check known categorical values.
    encoder = PREPROCESSOR.named_transformers_["encode"]

    for field, allowed in zip(
        CATEGORICAL_COLS,
        encoder.categories_,
    ):
        if row[field] not in allowed:
            raise ValueError(
                f"Unknown category for {field}: {row[field]}"
            )

    # Match the original training-time date correction.
    if (
        row["arrival_year"] == 2018
        and row["arrival_month"] == 2
        and row["arrival_date"] == 29
    ):
        row["arrival_date"] = 28

    try:
        arrival = date(
            row["arrival_year"],
            row["arrival_month"],
            row["arrival_date"],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid arrival date.") from exc

    row["arrival_dayofweek"] = arrival.weekday()
    row["arrival_day"] = arrival.day
    row["arrival_quarter"] = (arrival.month - 1) // 3 + 1
    row["is_weekend_arrival"] = int(arrival.weekday() >= 5)

    # Construct the exact features expected by the encoder.
    feature_columns = list(
        dict.fromkeys(CATEGORICAL_COLS + NUMERICAL_COLS)
    )

    return pd.DataFrame(
        [{field: row[field] for field in feature_columns}]
    )


# ============================================================
# 3. GENERATE PREDICTION
# ============================================================

def predict_reservation(reservation):
    """Predict cancellation probability for one reservation."""

    input_df = prepare_reservation(reservation)

    # Apply the fitted training-time preprocessing.
    transformed = PREPROCESSOR.transform(input_df)

    features = pd.DataFrame(
        transformed,
        columns=FEATURE_NAMES,
    )

    # Route the booking to its assigned cluster.
    cluster_id = int(KMEANS.predict(features)[0])

    model = MODELS[cluster_id]

    # Predict the cancellation probability.
    canceled_class = int(
        LABEL_ENCODER.transform(["Canceled"])[0]
    )

    class_index = list(model.classes_).index(canceled_class)

    cancellation_probability = float(
        model.predict_proba(features)[0][class_index]
    )

    prediction = (
        "Canceled"
        if cancellation_probability >= 0.5
        else "Not_Canceled"
    )

    return {
        "prediction": prediction,
        "cancellation_probability": round(
            cancellation_probability, 4
        ),
        "cancellation_risk_percent": round(
            cancellation_probability * 100, 2
        ),
        "cluster_id": cluster_id,
        "model_version": "1.0.0",
    }


# ============================================================
# 4. AWS LAMBDA HANDLER
# ============================================================

def lambda_handler(event, context):
    """AWS Lambda entry point for prediction requests."""

    try:
        # HTTP APIs often provide the JSON request as body.
        body = event.get("body", event)

        if isinstance(body, str):
            body = json.loads(body)

        if not isinstance(body, dict):
            raise ValueError("Request body must be a JSON object.")

        result = predict_reservation(body)

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json"
            },
            "body": json.dumps(result),
        }

    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        return {
            "statusCode": 400,
            "headers": {
                "Content-Type": "application/json"
            },
            "body": json.dumps({"error": str(exc)}),
        }

    except Exception:
        # Avoid exposing internal details to API clients.
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json"
            },
            "body": json.dumps({
                "error": "An internal prediction error occurred."
            }),
        }
