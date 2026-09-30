"""
STEP 12 - Final Test, Two-Stage Model and Model Saving

Final evaluation on the separate validation dataset.
Stage 1: predict BUILDINGID using kNN.
Stage 2: predict FLOOR using a Gradient Boosting model for the predicted building.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import joblib

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")

TRAIN_PATH = os.path.join(DATA_DIR, "trainingData.csv")
VALIDATION_PATH = os.path.join(DATA_DIR, "validationData.csv")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

# Building classifier
BUILDING_K = 1

# Final floor model
GB_N_ESTIMATORS = 100
GB_LEARNING_RATE = 0.1
GB_MAX_DEPTH = 4
GB_MIN_SAMPLES_LEAF = 2

# Dataset convention
MISSING_VALUE = 100
MISSING_FILL = -105

BUILDINGS = [0, 1, 2]


# ============================================================
# DATA LOADING
# ============================================================

def load_datasets():
    """
    Load trainingData.csv and validationData.csv.
    Returns the two dataframes.
    """

    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(
            f"Training file not found:\n{TRAIN_PATH}\n\n"
            "Make sure trainingData.csv is inside data/."
        )

    if not os.path.exists(VALIDATION_PATH):
        raise FileNotFoundError(
            f"Validation file not found:\n{VALIDATION_PATH}\n\n"
            "Make sure validationData.csv is inside data/."
        )

    train_df = pd.read_csv(TRAIN_PATH)
    validation_df = pd.read_csv(VALIDATION_PATH)

    return train_df, validation_df


# ============================================================
# WAP FEATURE HANDLING
# ============================================================

def get_wap_columns(df):
    """
    Return all WiFi fingerprint columns beginning with WAP.
    """

    wap_cols = [
        col for col in df.columns
        if str(col).startswith("WAP")
    ]

    if len(wap_cols) == 0:
        raise ValueError(
            "No WAP columns were found in the dataset."
        )

    return wap_cols


def select_active_waps(train_df, wap_cols):
    """
    Select WAP columns that are heard at least once in training data.
    """

    train_wap = train_df[wap_cols].copy()

    active_mask = (train_wap != MISSING_VALUE).any(axis=0)

    active_cols = list(train_wap.columns[active_mask])

    if len(active_cols) == 0:
        raise ValueError(
            "No active WAP features were found."
        )

    return active_cols


def clean_features(df, wap_cols):
    """
    Replace missing RSSI value 100 with -105.
    """

    X = df[wap_cols].copy()

    X = X.replace(MISSING_VALUE, MISSING_FILL)

    X = X.astype(np.float32)

    return X


# ============================================================
# BUILDING CLASSIFIER
# ============================================================

def train_building_model(X_train, y_building):
    """
    Train a standardized kNN classifier for BUILDINGID.
    """

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X_train)

    model = KNeighborsClassifier(
        n_neighbors=BUILDING_K,
        weights="uniform",
        metric="euclidean",
        n_jobs=-1
    )

    model.fit(X_scaled, y_building)

    return scaler, model


# ============================================================
# FLOOR MODELS
# ============================================================

def train_floor_model(X_train, y_floor):
    """
    Train Gradient Boosting classifier for floor prediction.
    """

    model = GradientBoostingClassifier(
        n_estimators=GB_N_ESTIMATORS,
        learning_rate=GB_LEARNING_RATE,
        max_depth=GB_MAX_DEPTH,
        min_samples_leaf=GB_MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE
    )

    model.fit(X_train, y_floor)

    return model


# ============================================================
# CONFUSION MATRIX TABLE
# ============================================================

def save_confusion_matrix(y_true, y_pred, building_id):
    """
    Save the floor confusion matrix for one building.
    """

    labels = sorted(
        set(y_true).union(set(y_pred))
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual_Floor_{x}" for x in labels],
        columns=[f"Predicted_Floor_{x}" for x in labels]
    )

    output_path = os.path.join(
        TABLES_DIR,
        f"final_floor_confusion_building_{building_id}.csv"
    )

    cm_df.to_csv(output_path)

    return cm_df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("STEP 12 - FINAL TEST + TWO-STAGE MODEL")
    print("=" * 60)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    print("\nLoading data...")

    train_df, validation_df = load_datasets()

    print(f"Training shape    : {train_df.shape}")
    print(f"Validation shape  : {validation_df.shape}")

    # --------------------------------------------------------
    # BASIC COLUMN CHECK
    # --------------------------------------------------------

    required_columns = [
        "BUILDINGID",
        "FLOOR"
    ]

    for col in required_columns:

        if col not in train_df.columns:
            raise ValueError(
                f"Training data missing required column: {col}"
            )

        if col not in validation_df.columns:
            raise ValueError(
                f"Validation data missing required column: {col}"
            )

    # --------------------------------------------------------
    # WAP SELECTION
    # --------------------------------------------------------

    print("\nSelecting active WAP features...")

    train_wap_cols = get_wap_columns(train_df)
    validation_wap_cols = get_wap_columns(validation_df)

    print(f"Training WAP columns   : {len(train_wap_cols)}")
    print(f"Validation WAP columns : {len(validation_wap_cols)}")

    # Only columns that exist in BOTH datasets
    common_waps = [
        col for col in train_wap_cols
        if col in validation_wap_cols
    ]

    print(f"Common WAP columns     : {len(common_waps)}")

    # Active columns are selected ONLY using training data
    active_waps = select_active_waps(
        train_df,
        common_waps
    )

    print(f"Active WAP features    : {len(active_waps)}")

    # --------------------------------------------------------
    # CLEAN FEATURES
    # --------------------------------------------------------

    print("\nCleaning RSSI values...")

    X_train = clean_features(
        train_df,
        active_waps
    )

    X_validation = clean_features(
        validation_df,
        active_waps
    )

    y_train_building = train_df["BUILDINGID"].astype(int)
    y_validation_building = validation_df["BUILDINGID"].astype(int)

    y_train_floor = train_df["FLOOR"].astype(int)
    y_validation_floor = validation_df["FLOOR"].astype(int)

    print("Cleaning complete.")
    print(f"Final feature matrix: {X_train.shape}")

    # ========================================================
    # STAGE 1 - BUILDING CLASSIFIER
    # ========================================================

    print("\n")
    print("=" * 60)
    print("STAGE 1 - BUILDING CLASSIFIER")
    print("=" * 60)

    print("\nTraining building kNN...")

    building_scaler, building_model = train_building_model(
        X_train,
        y_train_building
    )

    X_train_scaled = building_scaler.transform(X_train)
    X_validation_scaled = building_scaler.transform(
        X_validation
    )

    building_train_pred = building_model.predict(
        X_train_scaled
    )

    building_validation_pred = building_model.predict(
        X_validation_scaled
    )

    building_train_accuracy = accuracy_score(
        y_train_building,
        building_train_pred
    )

    building_validation_accuracy = accuracy_score(
        y_validation_building,
        building_validation_pred
    )

    print(
        f"\nBuilding training accuracy   : "
        f"{building_train_accuracy:.4f}"
    )

    print(
        f"Building validation accuracy : "
        f"{building_validation_accuracy:.4f}"
    )

    # --------------------------------------------------------
    # SAVE BUILDING MODEL
    # --------------------------------------------------------

    building_bundle = {
        "model": building_model,
        "scaler": building_scaler,
        "wap_columns": active_waps,
        "missing_fill": MISSING_FILL,
        "model_type": "KNN",
        "k": BUILDING_K,
    }

    building_model_path = os.path.join(
        MODELS_DIR,
        "building_knn.joblib"
    )

    joblib.dump(
        building_bundle,
        building_model_path
    )

    print(
        f"\nBuilding model saved:\n"
        f"{building_model_path}"
    )

    # ========================================================
    # STAGE 2 - FLOOR MODELS
    # ========================================================

    print("\n")
    print("=" * 60)
    print("STAGE 2 - FLOOR MODELS")
    print("=" * 60)

    floor_results = []

    floor_models = {}

    # Train one model for each building
    for building_id in BUILDINGS:

        print("\n" + "-" * 50)
        print(f"BUILDING {building_id}")
        print("-" * 50)

        # Training rows belonging to this building
        train_mask = (
            y_train_building == building_id
        )

        # Validation rows belonging to this building
        validation_mask = (
            y_validation_building == building_id
        )

        X_b_train = X_train.loc[train_mask]
        y_b_train = y_train_floor.loc[train_mask]

        X_b_validation = X_validation.loc[
            validation_mask
        ]

        y_b_validation = y_validation_floor.loc[
            validation_mask
        ]

        print(
            f"Training samples   : {len(X_b_train)}"
        )

        print(
            f"Validation samples : {len(X_b_validation)}"
        )

        print(
            f"Floors             : "
            f"{sorted(y_b_train.unique())}"
        )

        # Train
        print("\nTraining Gradient Boosting...")

        floor_model = train_floor_model(
            X_b_train,
            y_b_train
        )

        # Predict
        floor_pred = floor_model.predict(
            X_b_validation
        )

        floor_accuracy = accuracy_score(
            y_b_validation,
            floor_pred
        )

        floor_report = classification_report(
            y_b_validation,
            floor_pred,
            output_dict=True,
            zero_division=0
        )

        macro_f1 = floor_report[
            "macro avg"
        ]["f1-score"]

        error = 1.0 - floor_accuracy

        print(
            f"\nAccuracy : {floor_accuracy:.4f}"
        )

        print(
            f"Error    : {error:.4f}"
        )

        print(
            f"Macro-F1 : {macro_f1:.4f}"
        )

        # Save confusion matrix
        cm_df = save_confusion_matrix(
            y_b_validation,
            floor_pred,
            building_id
        )

        print("\nConfusion matrix:")

        print(cm_df)

        # Save model in memory
        floor_models[building_id] = floor_model

        # Save model to disk
        floor_model_path = os.path.join(
            MODELS_DIR,
            f"floor_building_{building_id}.joblib"
        )

        floor_bundle = {
            "model": floor_model,
            "wap_columns": active_waps,
            "missing_fill": MISSING_FILL,
            "building_id": building_id,
            "model_type": "GradientBoosting",
            "n_estimators": GB_N_ESTIMATORS,
            "learning_rate": GB_LEARNING_RATE,
            "max_depth": GB_MAX_DEPTH,
        }

        joblib.dump(
            floor_bundle,
            floor_model_path
        )

        print(
            f"\nModel saved:\n"
            f"{floor_model_path}"
        )

        floor_results.append({
            "Building": building_id,
            "Training Samples": len(X_b_train),
            "Validation Samples": len(X_b_validation),
            "Accuracy": floor_accuracy,
            "Error": error,
            "Macro-F1": macro_f1
        })

    # ========================================================
    # TWO-STAGE END-TO-END TEST
    # ========================================================

    print("\n")
    print("=" * 60)
    print("TWO-STAGE END-TO-END TEST")
    print("=" * 60)

    final_predictions = []
    final_actual = []

    for index in range(len(X_validation)):

        # ----------------------------------------------
        # Stage 1: Building
        # ----------------------------------------------

        x_row = X_validation.iloc[
            index:index + 1
        ]

        x_row_scaled = building_scaler.transform(
            x_row
        )

        predicted_building = int(
            building_model.predict(
                x_row_scaled
            )[0]
        )

        # ----------------------------------------------
        # Stage 2: Floor
        # ----------------------------------------------

        if predicted_building not in floor_models:

            # Safety fallback
            predicted_building = 0

        floor_model = floor_models[
            predicted_building
        ]

        predicted_floor = int(
            floor_model.predict(
                x_row
            )[0]
        )

        actual_building = int(
            y_validation_building.iloc[index]
        )

        actual_floor = int(
            y_validation_floor.iloc[index]
        )

        final_predictions.append(
            (predicted_building, predicted_floor)
        )

        final_actual.append(
            (actual_building, actual_floor)
        )

    # --------------------------------------------------------
    # END-TO-END METRICS
    # --------------------------------------------------------

    building_correct = 0
    floor_correct_given_building = 0
    complete_correct = 0

    for pred, actual in zip(
        final_predictions,
        final_actual
    ):

        pred_building, pred_floor = pred
        actual_building, actual_floor = actual

        if pred_building == actual_building:
            building_correct += 1

        if (
            pred_building == actual_building
            and pred_floor == actual_floor
        ):
            complete_correct += 1

        # Floor prediction using the actual building.
        # This is already measured separately above.
        if pred_floor == actual_floor:
            floor_correct_given_building += 1

    total = len(final_predictions)

    end_to_end_accuracy = (
        complete_correct / total
    )

    building_accuracy = (
        building_correct / total
    )

    floor_prediction_accuracy = (
        floor_correct_given_building / total
    )

    print(
        f"\nBuilding accuracy        : "
        f"{building_accuracy:.4f}"
    )

    print(
        f"Floor accuracy           : "
        f"{floor_prediction_accuracy:.4f}"
    )

    print(
        f"End-to-end accuracy      : "
        f"{end_to_end_accuracy:.4f}"
    )

    # ========================================================
    # SAVE FINAL RESULTS
    # ========================================================

    floor_results_df = pd.DataFrame(
        floor_results
    )

    floor_results_path = os.path.join(
        TABLES_DIR,
        "final_floor_results.csv"
    )

    floor_results_df.to_csv(
        floor_results_path,
        index=False
    )

    # Save end-to-end result
    summary_df = pd.DataFrame([{
        "Building Accuracy":
            building_accuracy,
        "Floor Accuracy":
            floor_prediction_accuracy,
        "End-to-End Accuracy":
            end_to_end_accuracy
    }])

    summary_path = os.path.join(
        TABLES_DIR,
        "final_two_stage_results.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    # ========================================================
    # SAVE COMPLETE CONFIGURATION
    # ========================================================

    config = {
        "active_wap_columns": active_waps,
        "missing_value_original": MISSING_VALUE,
        "missing_value_replacement": MISSING_FILL,
        "building_model": "KNN",
        "building_k": BUILDING_K,
        "floor_model": "GradientBoosting",
        "gb_n_estimators": GB_N_ESTIMATORS,
        "gb_learning_rate": GB_LEARNING_RATE,
        "gb_max_depth": GB_MAX_DEPTH,
        "random_state": RANDOM_STATE,
    }

    config_path = os.path.join(
        MODELS_DIR,
        "model_config.joblib"
    )

    joblib.dump(
        config,
        config_path
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print("\n")
    print("=" * 60)
    print("FINAL STEP 12 RESULTS")
    print("=" * 60)

    print("\nPer-building floor results:")
    print(floor_results_df.to_string(index=False))

    print("\nTwo-stage results:")
    print(summary_df.to_string(index=False))

    print("\n")
    print("=" * 60)
    print("FILES SAVED")
    print("=" * 60)

    print(
        f"\nBuilding model:\n"
        f"{building_model_path}"
    )

    for building_id in BUILDINGS:

        print(
            f"\nFloor model Building {building_id}:\n"
            f"{os.path.join(MODELS_DIR, f'floor_building_{building_id}.joblib')}"
        )

    print(
        f"\nFloor results:\n"
        f"{floor_results_path}"
    )

    print(
        f"\nTwo-stage results:\n"
        f"{summary_path}"
    )

    print(
        f"\nModel configuration:\n"
        f"{config_path}"
    )

    print("\n")
    print("=" * 60)
    print("STEP 12 FINAL TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()