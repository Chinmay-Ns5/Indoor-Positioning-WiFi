"""
Indoor Positioning - Live Two-Stage Demo

Stage 1:
    Predict BUILDINGID using the saved building classifier.

Stage 2:
    Predict FLOOR using the saved floor classifier for that building.

The demo uses random fingerprints from validationData.csv.
"""

import os
import sys
import random
import warnings

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


TRAIN_PATH = os.path.join(
    DATA_DIR,
    "trainingData.csv"
)

VALIDATION_PATH = os.path.join(
    DATA_DIR,
    "validationData.csv"
)

BUILDING_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "building_knn.joblib"
)


# ============================================================
# CONFIGURATION
# ============================================================

NUM_EXAMPLES = 5
MISSING_VALUE = -105

BUILDINGS = [0, 1, 2]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_validation_data():
    """
    Load validationData.csv and return the dataframe.
    """

    if not os.path.exists(VALIDATION_PATH):
        raise FileNotFoundError(
            f"Validation data not found:\n{VALIDATION_PATH}"
        )

    df = pd.read_csv(VALIDATION_PATH)

    return df


def load_training_data():
    """
    Load trainingData.csv for determining the available WAP columns.
    """

    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(
            f"Training data not found:\n{TRAIN_PATH}"
        )

    df = pd.read_csv(TRAIN_PATH)

    return df


def get_wap_columns(df):
    """
    Return all WAP/RSSI feature columns from a dataframe.
    """

    return [
        col
        for col in df.columns
        if str(col).startswith("WAP")
    ]


def clean_rssi(X):
    """
    Replace the UJIndoorLoc missing-signal value 100 with -105.
    Return a writable floating-point NumPy array.
    """

    # IMPORTANT:
    # np.asarray() alone can preserve read-only memory.
    # .copy() guarantees that assignment is allowed.
    X = np.asarray(X, dtype=np.float64).copy()

    X[X == 100] = MISSING_VALUE

    return X


def prepare_fingerprint(row, expected_waps):
    """
    Prepare one validation fingerprint using the exact WAP
    feature ordering expected by a saved model.
    """

    values = []

    for wap in expected_waps:

        if wap in row.index:
            value = row[wap]

            # Handle NaN/missing values.
            if pd.isna(value):
                value = MISSING_VALUE

            values.append(float(value))

        else:
            # If a WAP is absent from the validation dataframe,
            # treat it as not heard.
            values.append(float(MISSING_VALUE))

    X = np.array(
        values,
        dtype=np.float64
    ).reshape(1, -1)

    # Make absolutely sure it is writable.
    X = X.copy()

    X[X == 100] = MISSING_VALUE

    return X


def extract_model(bundle, name="model"):
    """
    Extract the sklearn estimator from a saved joblib bundle.

    Supports both:
        1. raw sklearn models
        2. dictionaries containing a 'model' key
    """

    # Normal sklearn model
    if hasattr(bundle, "predict"):
        return bundle

    # Saved dictionary bundle
    if isinstance(bundle, dict):

        if "model" in bundle:

            model = bundle["model"]

            if hasattr(model, "predict"):
                return model

            raise TypeError(
                f"{name} bundle contains a 'model' entry "
                f"but it has no predict() method."
            )

    raise TypeError(
        f"Could not extract a prediction model from {name}.\n"
        f"Loaded object type: {type(bundle)}"
    )


def extract_wap_columns(bundle, fallback_waps):
    """
    Extract the WAP feature list stored with a model bundle.
    """

    if isinstance(bundle, dict):

        waps = bundle.get("wap_columns")

        if waps is not None and len(waps) > 0:
            return list(waps)

    # Fallback for older model files.
    return list(fallback_waps)


def extract_scaler(bundle):
    """
    Extract an optional scaler from a saved model bundle.
    """

    if isinstance(bundle, dict):

        scaler = bundle.get("scaler")

        if scaler is not None:
            return scaler

    return None


def get_confidence(model, X):
    """
    Return the maximum class probability when predict_proba()
    is available. Otherwise return None.
    """

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(X)

        if probabilities is not None:
            return float(
                np.max(probabilities[0])
            )

    return None


def load_floor_models():
    """
    Load the saved floor-model bundles for all three buildings.
    """

    floor_models = {}

    for building in BUILDINGS:

        path = os.path.join(
            MODELS_DIR,
            f"floor_building_{building}.joblib"
        )

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"Floor model for Building {building} "
                f"not found:\n{path}"
            )

        floor_models[building] = joblib.load(path)

    return floor_models


def predict_location(
    row,
    building_bundle,
    floor_bundles,
    fallback_waps
):
    """
    Perform the complete two-stage prediction.

    Stage 1:
        predict building.

    Stage 2:
        use that building's floor model to predict floor.

    Returns:
        predicted_building,
        predicted_floor,
        building_confidence,
        floor_confidence
    """

    # ========================================================
    # STAGE 1 - BUILDING
    # ========================================================

    building_model = extract_model(
        building_bundle,
        "building model"
    )

    building_waps = extract_wap_columns(
        building_bundle,
        fallback_waps
    )

    X_building = prepare_fingerprint(
        row,
        building_waps
    )

    building_scaler = extract_scaler(
        building_bundle
    )

    if building_scaler is not None:

        X_building = building_scaler.transform(
            X_building
        )

    predicted_building = building_model.predict(
        X_building
    )[0]

    predicted_building = int(predicted_building)

    building_confidence = get_confidence(
        building_model,
        X_building
    )

    # ========================================================
    # STAGE 2 - FLOOR
    # ========================================================

    if predicted_building not in floor_bundles:

        raise ValueError(
            f"No floor model available for "
            f"predicted Building {predicted_building}"
        )

    floor_bundle = floor_bundles[
        predicted_building
    ]

    floor_model = extract_model(
        floor_bundle,
        f"floor model B{predicted_building}"
    )

    floor_waps = extract_wap_columns(
        floor_bundle,
        fallback_waps
    )

    X_floor = prepare_fingerprint(
        row,
        floor_waps
    )

    floor_scaler = extract_scaler(
        floor_bundle
    )

    if floor_scaler is not None:

        X_floor = floor_scaler.transform(
            X_floor
        )

    predicted_floor = floor_model.predict(
        X_floor
    )[0]

    # Convert NumPy integer to normal Python integer.
    try:
        predicted_floor = int(predicted_floor)
    except (TypeError, ValueError):
        predicted_floor = predicted_floor

    floor_confidence = get_confidence(
        floor_model,
        X_floor
    )

    return (
        predicted_building,
        predicted_floor,
        building_confidence,
        floor_confidence
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("INDOOR POSITIONING - LIVE DEMO")
    print("=" * 60)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print("\nLoading validation data...")

    validation_df = load_validation_data()

    print(
        f"Validation samples : {len(validation_df)}"
    )

    # --------------------------------------------------------
    # DETERMINE WAP FEATURES
    # --------------------------------------------------------

    print(
        "\nLoading training data for active WAP selection..."
    )

    training_df = load_training_data()

    training_waps = get_wap_columns(
        training_df
    )

    validation_waps = get_wap_columns(
        validation_df
    )

    common_waps = [
        wap
        for wap in training_waps
        if wap in validation_waps
    ]

    print(
        f"Training WAP columns   : {len(training_waps)}"
    )

    print(
        f"Validation WAP columns : {len(validation_waps)}"
    )

    print(
        f"Common WAP columns     : {len(common_waps)}"
    )

    # --------------------------------------------------------
    # LOAD BUILDING MODEL
    # --------------------------------------------------------

    print("\nLoading trained models...")

    if not os.path.exists(BUILDING_MODEL_PATH):

        raise FileNotFoundError(
            "Building model not found:\n"
            f"{BUILDING_MODEL_PATH}"
        )

    building_bundle = joblib.load(
        BUILDING_MODEL_PATH
    )

    # This is intentionally NOT:
    #
    # building_model = joblib.load(...)
    #
    # because the file contains a dictionary bundle.

    building_model = extract_model(
        building_bundle,
        "building model"
    )

    print(
        f"Building model          : Loaded "
        f"({type(building_model).__name__})"
    )

    floor_bundles = load_floor_models()

    for building in BUILDINGS:

        floor_model = extract_model(
            floor_bundles[building],
            f"floor model B{building}"
        )

        print(
            f"Floor model B{building}          : Loaded "
            f"({type(floor_model).__name__})"
        )

    # --------------------------------------------------------
    # DEMO INFORMATION
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("TWO-STAGE PREDICTION")
    print("=" * 60)

    print("\nStage 1 : Predict Building")
    print("Stage 2 : Predict Floor inside Building")

    # --------------------------------------------------------
    # RANDOM EXAMPLES
    # --------------------------------------------------------

    print(
        f"\nRunning {NUM_EXAMPLES} random examples..."
    )

    rng = random.Random(42)

    sample_count = min(
        NUM_EXAMPLES,
        len(validation_df)
    )

    sample_indices = rng.sample(
        range(len(validation_df)),
        sample_count
    )

    successful = 0

    # --------------------------------------------------------
    # RUN DEMO
    # --------------------------------------------------------

    for example_number, index in enumerate(
        sample_indices,
        start=1
    ):

        row = validation_df.iloc[index]

        print("\n" + "-" * 60)
        print(f"EXAMPLE {example_number}")
        print("-" * 60)

        try:

            (
                predicted_building,
                predicted_floor,
                building_confidence,
                floor_confidence
            ) = predict_location(
                row,
                building_bundle,
                floor_bundles,
                common_waps
            )

            actual_building = int(
                row["BUILDINGID"]
            )

            actual_floor = int(
                row["FLOOR"]
            )

            building_correct = (
                predicted_building
                == actual_building
            )

            floor_correct = (
                predicted_floor
                == actual_floor
            )

            end_to_end_correct = (
                building_correct
                and floor_correct
            )

            print(
                f"Actual location     : "
                f"Building {actual_building}, "
                f"Floor {actual_floor}"
            )

            print(
                f"Predicted location   : "
                f"Building {predicted_building}, "
                f"Floor {predicted_floor}"
            )

            if building_confidence is not None:

                print(
                    f"Building confidence : "
                    f"{building_confidence:.4f}"
                )

            else:

                print(
                    "Building confidence : N/A"
                )

            if floor_confidence is not None:

                print(
                    f"Floor confidence    : "
                    f"{floor_confidence:.4f}"
                )

            else:

                print(
                    "Floor confidence    : N/A"
                )

            print(
                f"Building correct    : "
                f"{'YES' if building_correct else 'NO'}"
            )

            print(
                f"Floor correct       : "
                f"{'YES' if floor_correct else 'NO'}"
            )

            print(
                f"End-to-end correct  : "
                f"{'YES' if end_to_end_correct else 'NO'}"
            )

            successful += 1

        except Exception as error:

            print(
                f"EXAMPLE {example_number} FAILED"
            )

            print(
                f"Error: {type(error).__name__}: {error}"
            )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("LIVE DEMO COMPLETE")
    print("=" * 60)

    print(
        f"\nSuccessful predictions : "
        f"{successful}/{sample_count}"
    )

    if successful == sample_count:

        print(
            "Demo status : ALL PREDICTIONS COMPLETED"
        )

    elif successful > 0:

        print(
            "Demo status : SOME PREDICTIONS FAILED"
        )

    else:

        print(
            "Demo status : ALL PREDICTIONS FAILED"
        )


if __name__ == "__main__":
    main()