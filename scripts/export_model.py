
from pathlib import Path

import joblib
import pandas as pd

from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "Hotel Reservations.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "guest_shield_model.joblib"


# ============================================================
# 2. FEATURE DEFINITIONS
# ============================================================

CATEGORICAL_COLS = [
    "type_of_meal_plan",
    "room_type_reserved",
    "market_segment_type",
    "required_car_parking_space",
    "repeated_guest",
]

DISCRETE_NUMERICAL_COLS = [
    "no_of_adults",
    "no_of_children",
    "no_of_weekend_nights",
    "no_of_week_nights",
    "arrival_year",
    "arrival_month",
    "arrival_dayofweek",
    "arrival_day",
    "arrival_quarter",
    "is_weekend_arrival",
    "no_of_previous_cancellations",
    "no_of_previous_bookings_not_canceled",
    "no_of_special_requests",
]

CONTINUOUS_NUMERICAL_COLS = [
    "lead_time",
    "avg_price_per_room",
]

NUMERICAL_COLS = (
    CONTINUOUS_NUMERICAL_COLS + DISCRETE_NUMERICAL_COLS
)


# ============================================================
# 3. LOAD DATA
# ============================================================

def load_data():
    """Load the original hotel reservation dataset."""

    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully!")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    return df


# ============================================================
# 4. DATA CLEANING AND FEATURE ENGINEERING
# ============================================================

def prepare_data(df):
    """Clean reservation data and create arrival-date features."""

    df = df.copy()

    # Booking ID is not a predictive feature.
    df = df.drop(columns=["Booking_ID"])

    # Fix invalid February 29 dates in 2018.
    invalid_feb29 = (
        (df["arrival_year"] == 2018)
        & (df["arrival_month"] == 2)
        & (df["arrival_date"] == 29)
    )

    print(
        "Invalid February 29 dates corrected: "
        f"{invalid_feb29.sum()}"
    )

    df.loc[invalid_feb29, "arrival_date"] = 28

    # Build arrival dates.
    arrival_date = pd.to_datetime(
        df["arrival_year"].astype(str)
        + "-"
        + df["arrival_month"].astype(str).str.zfill(2)
        + "-"
        + df["arrival_date"].astype(str).str.zfill(2),
        errors="coerce",
    )

    # Engineer additional arrival-date features.
    df["arrival_dayofweek"] = arrival_date.dt.dayofweek
    df["arrival_day"] = arrival_date.dt.day
    df["arrival_quarter"] = arrival_date.dt.quarter

    df["is_weekend_arrival"] = (
        df["arrival_dayofweek"].isin([5, 6]).astype(int)
    )

    # Original day-of-month column is replaced by arrival_day.
    df = df.drop(columns=["arrival_date"])

    print(f"Prepared dataset shape: {df.shape}")

    return df


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

def split_data(df):
    """Create an 80/20 stratified training and testing split."""

    train, test = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["booking_status"],
    )

    X_train = train.drop(columns=["booking_status"])
    y_train = train["booking_status"]

    X_test = test.drop(columns=["booking_status"])
    y_test = test["booking_status"]

    print("\nTrain/test split completed!")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")

    return X_train, X_test, y_train, y_test


# ============================================================
# 6. FEATURE PREPROCESSING
# ============================================================

def preprocess_features(X_train, X_test):
    """Fit the encoder on training data only."""

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "encode",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_COLS,
            ),
            (
                "passthrough",
                "passthrough",
                NUMERICAL_COLS,
            ),
        ]
    )

    X_train_array = preprocessor.fit_transform(X_train)
    X_test_array = preprocessor.transform(X_test)

    encoded_columns = list(
        preprocessor.named_transformers_[
            "encode"
        ].get_feature_names_out(CATEGORICAL_COLS)
    )

    all_feature_names = encoded_columns + NUMERICAL_COLS

    X_train_processed = pd.DataFrame(
        X_train_array,
        columns=all_feature_names,
    )

    X_test_processed = pd.DataFrame(
        X_test_array,
        columns=all_feature_names,
    )

    print("\nFeature preprocessing completed!")
    print(f"Training shape: {X_train_processed.shape}")
    print(f"Testing shape: {X_test_processed.shape}")

    return (
        X_train_processed,
        X_test_processed,
        preprocessor,
        all_feature_names,
    )


# ============================================================
# 7. ENCODE TARGET VARIABLE
# ============================================================

def encode_target(y_train, y_test):
    """Encode cancellation status as numeric classes."""

    label_encoder = LabelEncoder()

    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    print("\nTarget encoding:")

    for label, encoded in zip(
        label_encoder.classes_,
        label_encoder.transform(label_encoder.classes_),
    ):
        print(f"{label} = {encoded}")

    return (
        y_train_encoded,
        y_test_encoded,
        label_encoder,
    )


# ============================================================
# 8. BALANCE TRAINING DATA WITH SMOTE
# ============================================================

def balance_training_data(X_train, y_train):
    """Apply SMOTE to the training data only."""

    print("\nApplying SMOTE...")

    print("Class counts before SMOTE:")
    print(pd.Series(y_train).value_counts().sort_index())

    smote = SMOTE(random_state=42)

    X_resampled, y_resampled = smote.fit_resample(
        X_train,
        y_train,
    )

    print("\nClass counts after SMOTE:")
    print(pd.Series(y_resampled).value_counts().sort_index())

    print(f"Resampled training shape: {X_resampled.shape}")

    return X_resampled, y_resampled


# ============================================================
# 9. KMEANS CLUSTERING
# ============================================================

def train_clustering_model(X_train_resampled, X_test):
    """Train KMeans on resampled training data."""

    print("\nTraining KMeans clustering model...")

    kmeans = KMeans(
        n_clusters=2,
        random_state=42,
        n_init=10,
    )

    train_clusters = kmeans.fit_predict(X_train_resampled)
    test_clusters = kmeans.predict(X_test)

    print("\nTraining cluster distribution:")
    print(
        pd.Series(train_clusters)
        .value_counts()
        .sort_index()
    )

    print("\nTesting cluster distribution:")
    print(
        pd.Series(test_clusters)
        .value_counts()
        .sort_index()
    )

    return kmeans, train_clusters, test_clusters


# ============================================================
# 10. TRAIN CLUSTER-SPECIFIC RANDOM FOREST MODELS
# ============================================================

def train_random_forest_models(
    X_train_resampled,
    y_train_resampled,
    train_clusters,
):
    """Train one Random Forest classifier for each cluster."""

    models = {}

    for cluster_id in range(2):

        print(f"\nTraining Random Forest for cluster {cluster_id}...")

        cluster_mask = train_clusters == cluster_id

        X_cluster = X_train_resampled.loc[cluster_mask]
        y_cluster = y_train_resampled[cluster_mask]

        print(f"Training records: {len(X_cluster)}")

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_cluster, y_cluster)

        models[cluster_id] = model

        print(f"Cluster {cluster_id} model trained!")

    return models


# ============================================================
# 11. EVALUATE ON HELD-OUT TEST DATA
# ============================================================

def evaluate_models(
    models,
    X_test,
    y_test,
    test_clusters,
    label_encoder,
):
    """Evaluate the complete cluster-routing prediction system."""

    predictions = []

    for row_index, cluster_id in enumerate(test_clusters):

        # Select the model assigned to this booking's cluster.
        model = models[int(cluster_id)]

        # Predict the booking outcome.
        row = X_test.iloc[[row_index]]
        prediction = model.predict(row)[0]

        predictions.append(prediction)

    accuracy = accuracy_score(y_test, predictions)

    print("\n========== MODEL EVALUATION ==========")
    print(f"Overall test accuracy: {accuracy:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=label_encoder.classes_,
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    return accuracy


# ============================================================
# 12. EXPORT TRAINED MODEL BUNDLE
# ============================================================

def export_model(
    preprocessor,
    kmeans,
    models,
    label_encoder,
    all_feature_names,
    accuracy,
):
    """Save all inference components in one model bundle."""

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_bundle = {
        "preprocessor": preprocessor,
        "kmeans": kmeans,
        "models": models,
        "label_encoder": label_encoder,
        "feature_names": all_feature_names,
        "categorical_columns": CATEGORICAL_COLS,
        "numerical_columns": NUMERICAL_COLS,
        "test_accuracy": float(accuracy),
    }

    joblib.dump(
        model_bundle,
        MODEL_PATH,
        compress=3,
    )

    print("\n========== MODEL EXPORT ==========")
    print(f"Model saved to: {MODEL_PATH}")

    size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    print(f"Model size: {size_mb:.2f} MB")

    return MODEL_PATH


# ============================================================
# 13. MAIN EXECUTION
# ============================================================

def main():

    # Load and clean the original dataset.
    df = load_data()

    print("\nStarting data preparation...")
    df = prepare_data(df)

    # Reject unexpected missing values.
    missing_count = int(df.isnull().sum().sum())

    print(f"\nTotal missing values: {missing_count}")

    if missing_count > 0:
        raise ValueError(
            "Missing values detected. Resolve them before training."
        )

    # Split the data.
    X_train, X_test, y_train, y_test = split_data(df)

    # Encode categorical features.
    (
        X_train_processed,
        X_test_processed,
        preprocessor,
        all_feature_names,
    ) = preprocess_features(X_train, X_test)

    # Encode the target.
    (
        y_train_encoded,
        y_test_encoded,
        label_encoder,
    ) = encode_target(y_train, y_test)

    # Balance training classes.
    (
        X_train_resampled,
        y_train_resampled,
    ) = balance_training_data(
        X_train_processed,
        y_train_encoded,
    )

    # Train clustering model.
    (
        kmeans,
        train_clusters,
        test_clusters,
    ) = train_clustering_model(
        X_train_resampled,
        X_test_processed,
    )

    # Train cluster-specific classifiers.
    models = train_random_forest_models(
        X_train_resampled,
        y_train_resampled,
        train_clusters,
    )

    # Evaluate the complete prediction system.
    accuracy = evaluate_models(
        models,
        X_test_processed,
        y_test_encoded,
        test_clusters,
        label_encoder,
    )

    # Save the inference components.
    export_model(
        preprocessor,
        kmeans,
        models,
        label_encoder,
        all_feature_names,
        accuracy,
    )

    print("\nGuest Shield model training completed successfully!")


if __name__ == "__main__":
    main()
