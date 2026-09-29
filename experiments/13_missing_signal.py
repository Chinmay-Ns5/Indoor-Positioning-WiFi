"""
STEP 11 - MISSING SIGNAL ROBUSTNESS

Tests how kNN and Gradient Boosting behave when RSSI signals
are randomly hidden from the validation/test data.

Missing signal representation:
    100  -> not detected in raw dataset
    -105 -> cleaned missing signal representation

Experiment:
    0%, 10%, 20%, 30%, 40%, 50% signals hidden
"""

import os
import sys

import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORT DATA UTILITIES
# ============================================================

from src.data_utils import (
    load_data,
    get_active_wap_columns,
    split_by_building,
)


# ============================================================
# CONFIGURATION
# ============================================================

# IMPORTANT:
# The project uses these filenames according to the project plan.
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

TRAIN_PATH = os.path.join(
    DATA_DIR,
    "trainingData.csv"
)

VALIDATION_PATH = os.path.join(
    DATA_DIR,
    "validationData.csv"
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "tables"
)

FIGURES_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "figures"
)

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

BUILDING_ID = 0

KNN_K = 1

HIDDEN_FRACTIONS = [
    0.0,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
]

RANDOM_SEED = 42


# ============================================================
# DROP / HIDE SIGNALS
# ============================================================

def drop_signals(X, frac, rng):
    """
    Randomly hide a fraction of currently detected RSSI signals.

    Only signals with RSSI != -105 are eligible to be hidden.

    Hidden signals are represented as -105.
    """

    X_hidden = X.copy()

    values = X_hidden.to_numpy(copy=True)

    # Only genuinely detected signals can be hidden.
    detected_mask = values != -105

    random_values = rng.random(values.shape)

    hide_mask = (
        detected_mask
        & (random_values < frac)
    )

    values[hide_mask] = -105

    X_hidden = pd.DataFrame(
        values,
        columns=X.columns,
        index=X.index
    )

    return X_hidden


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, X_test, y_test):
    """
    Predict validation labels and calculate:
        Accuracy
        Error
        Macro-F1
    """

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    error = 1.0 - accuracy

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro"
    )

    return accuracy, error, macro_f1


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("STEP 11 - MISSING SIGNAL ROBUSTNESS")
    print("=" * 60)

    # --------------------------------------------------------
    # CHECK DATA PATHS BEFORE DOING ANYTHING
    # --------------------------------------------------------

    print("\nChecking dataset paths...")

    print(f"Training data:")
    print(TRAIN_PATH)

    print(f"\nValidation data:")
    print(VALIDATION_PATH)

    if not os.path.isfile(TRAIN_PATH):
        raise FileNotFoundError(
            "\nTraining dataset not found.\n"
            f"Expected:\n{TRAIN_PATH}\n\n"
            "Make sure the file is named exactly:\n"
            "trainingData.csv\n"
        )

    if not os.path.isfile(VALIDATION_PATH):
        raise FileNotFoundError(
            "\nValidation dataset not found.\n"
            f"Expected:\n{VALIDATION_PATH}\n\n"
            "Make sure the file is named exactly:\n"
            "validationData.csv\n"
        )

    print("\nDataset files found.")


    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print("\nLoading data...")

    train_df, validation_df = load_data(
        TRAIN_PATH,
        VALIDATION_PATH
    )

    print(
        f"Training shape: {train_df.shape}"
    )

    print(
        f"Validation shape: {validation_df.shape}"
    )


    # --------------------------------------------------------
    # ACTIVE WAP SELECTION
    # --------------------------------------------------------

    print("\nSelecting active WAPs...")

    active_waps = get_active_wap_columns(
        train_df
    )

    print(
        f"Active WAP features: {len(active_waps)}"
    )


    # --------------------------------------------------------
    # SPLIT BY BUILDING
    # --------------------------------------------------------

    print(
        f"\nSelecting Building {BUILDING_ID}..."
    )

    X_train, y_train = split_by_building(
        train_df,
        active_waps,
        BUILDING_ID,
        fill=-105
    )

    X_validation, y_validation = split_by_building(
        validation_df,
        active_waps,
        BUILDING_ID,
        fill=-105
    )

    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Validation samples: {len(X_validation)}"
    )

    print(
        f"Features: {X_train.shape[1]}"
    )


    # --------------------------------------------------------
    # REMOVE ANY INVALID VALUES
    # --------------------------------------------------------

    X_train = X_train.astype(float)
    X_validation = X_validation.astype(float)


    # --------------------------------------------------------
    # TRAIN KNN
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("TRAINING kNN")
    print("=" * 60)

    print(f"\nk = {KNN_K}")
    print(f"Training samples = {len(X_train)}")

    knn = KNeighborsClassifier(
        n_neighbors=KNN_K,
        metric="euclidean",
        n_jobs=-1
    )

    knn.fit(
        X_train,
        y_train
    )

    print("kNN training complete.")


    # --------------------------------------------------------
    # TRAIN GRADIENT BOOSTING
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("TRAINING GRADIENT BOOSTING")
    print("=" * 60)

    print(
        f"\nTraining samples = {len(X_train)}"
    )

    gb = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        random_state=RANDOM_SEED
    )

    gb.fit(
        X_train,
        y_train
    )

    print(
        "Gradient Boosting training complete."
    )


    # --------------------------------------------------------
    # RANDOM NUMBER GENERATOR
    # --------------------------------------------------------

    rng = np.random.default_rng(
        RANDOM_SEED
    )


    # --------------------------------------------------------
    # RUN MISSING SIGNAL EXPERIMENT
    # --------------------------------------------------------

    results = []

    print("\n" + "=" * 60)
    print("MISSING SIGNAL EXPERIMENT")
    print("=" * 60)

    for frac in HIDDEN_FRACTIONS:

        percentage = int(frac * 100)

        print(
            f"\nTesting {percentage}% hidden signals..."
        )

        # Hide signals ONLY in validation data.
        X_test_hidden = drop_signals(
            X_validation,
            frac,
            rng
        )

        # ----------------------------------------------------
        # kNN
        # ----------------------------------------------------

        knn_accuracy, knn_error, knn_f1 = evaluate_model(
            knn,
            X_test_hidden,
            y_validation
        )

        print("\nkNN:")
        print(
            f"  Accuracy : {knn_accuracy:.4f}"
        )
        print(
            f"  Error    : {knn_error:.4f}"
        )
        print(
            f"  Macro-F1 : {knn_f1:.4f}"
        )


        results.append({
            "Model": "kNN",
            "Hidden_Percentage": percentage,
            "Accuracy": knn_accuracy,
            "Error": knn_error,
            "Macro-F1": knn_f1
        })


        # ----------------------------------------------------
        # Gradient Boosting
        # ----------------------------------------------------

        gb_accuracy, gb_error, gb_f1 = evaluate_model(
            gb,
            X_test_hidden,
            y_validation
        )

        print("\nGradient Boosting:")
        print(
            f"  Accuracy : {gb_accuracy:.4f}"
        )
        print(
            f"  Error    : {gb_error:.4f}"
        )
        print(
            f"  Macro-F1 : {gb_f1:.4f}"
        )


        results.append({
            "Model": "Gradient Boosting",
            "Hidden_Percentage": percentage,
            "Accuracy": gb_accuracy,
            "Error": gb_error,
            "Macro-F1": gb_f1
        })


    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    output_path = os.path.join(
        RESULTS_DIR,
        "missing_signal_results.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )


    # --------------------------------------------------------
    # PRINT FINAL RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL MISSING SIGNAL RESULTS")
    print("=" * 60)

    print()

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )


    # --------------------------------------------------------
    # BEST RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("BEST RESULTS")
    print("=" * 60)

    for model_name in [
        "kNN",
        "Gradient Boosting"
    ]:

        model_results = results_df[
            results_df["Model"] == model_name
        ]

        best_row = model_results.loc[
            model_results["Accuracy"].idxmax()
        ]

        print(
            f"\n{model_name}:"
        )

        print(
            f"  Best accuracy: "
            f"{best_row['Accuracy']:.4f}"
        )

        print(
            f"  Hidden signals: "
            f"{int(best_row['Hidden_Percentage'])}%"
        )


    # --------------------------------------------------------
    # SAVE SIMPLE PLOT
    # --------------------------------------------------------

    try:

        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))

        for model_name in [
            "kNN",
            "Gradient Boosting"
        ]:

            model_results = results_df[
                results_df["Model"] == model_name
            ]

            plt.plot(
                model_results["Hidden_Percentage"],
                model_results["Accuracy"],
                marker="o",
                label=model_name
            )

        plt.xlabel(
            "Percentage of Signals Hidden"
        )

        plt.ylabel(
            "Accuracy"
        )

        plt.title(
            "Model Robustness to Missing RSSI Signals"
        )

        plt.legend()

        plt.grid(True)

        plt.tight_layout()

        plot_path = os.path.join(
            FIGURES_DIR,
            "missing_signal_robustness.png"
        )

        plt.savefig(
            plot_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"\nPlot saved to:\n{plot_path}"
        )

    except Exception as e:

        print(
            f"\nWarning: could not create plot: {e}"
        )


    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FILES SAVED")
    print("=" * 60)

    print(
        f"\nResults:"
        f"\n{output_path}"
    )

    print("\n" + "=" * 60)
    print("MISSING SIGNAL EXPERIMENT COMPLETE")
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()