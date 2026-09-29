"""
Gradient Boosting Experiments
Indoor Positioning Using WiFi Fingerprints

Experiments:
1. Find best number of boosting iterations
2. Feature importance
3. Partial dependence of most important WAP
4. Per-floor error
5. Small-data experiment using 100 samples
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.inspection import PartialDependenceDisplay

from src.data_utils import load_data, get_active_wap_columns


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data" / "trainingData.csv"
VALIDATION_PATH = ROOT / "data" / "validationData.csv"

RESULTS_DIR = ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

BUILDING_ID = 0

N_ESTIMATORS = 300
MAX_DEPTH = 4
LEARNING_RATE = 0.1

RANDOM_STATE = 0

SAMPLE_SIZES = [100, 1000, 2000]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_rssi_dataframe(df, wap_cols):
    """
    Extract WAP features and replace RSSI value 100
    ('not detected') with -105.
    """

    X = df[wap_cols].copy()

    # IMPORTANT:
    # 100 means the WAP was not detected.
    # Replace it with a very weak signal.
    X = X.replace(100, -105)

    return X


def prepare_building_data(df, wap_cols, building_id):
    """
    Extract one building directly.

    We intentionally do NOT use split_by_building()
    here because the current data_utils implementation
    returns X/y rather than train/test dataframes.
    """

    building_df = df[df["BUILDINGID"] == building_id].copy()

    X = clean_rssi_dataframe(building_df, wap_cols)
    y = building_df["FLOOR"].copy()

    return X, y


def train_gradient_boosting(X_train, y_train):
    """
    Create and train Gradient Boosting classifier.
    """

    model = GradientBoostingClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)

    return model


def find_best_iterations(model, X_val, y_val):
    """
    Evaluate staged predictions to find the number of
    trees with the lowest validation error.
    """

    errors = []
    accuracies = []

    for i, predictions in enumerate(
        model.staged_predict(X_val),
        start=1
    ):
        accuracy = accuracy_score(y_val, predictions)
        error = 1.0 - accuracy

        accuracies.append(accuracy)
        errors.append(error)

    best_index = int(np.argmin(errors))

    best_n = best_index + 1
    best_accuracy = accuracies[best_index]
    best_error = errors[best_index]

    return (
        best_n,
        best_accuracy,
        best_error,
        errors,
        accuracies
    )


def evaluate_model(model, X_test, y_test):
    """
    Calculate accuracy, error and macro-F1.
    """

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
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
    print("GRADIENT BOOSTING EXPERIMENTS")
    print("=" * 60)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print("\nLoading data...")

    train_df, validation_df = load_data(
        TRAIN_PATH,
        VALIDATION_PATH
    )

    print(f"Training shape: {train_df.shape}")
    print(f"Validation shape: {validation_df.shape}")

    # --------------------------------------------------------
    # ACTIVE WAPS
    # --------------------------------------------------------

    print("\nSelecting active WAPs...")

    active_waps = get_active_wap_columns(train_df)

    print(f"Active WAP features: {len(active_waps)}")

    # --------------------------------------------------------
    # BUILDING 0
    # --------------------------------------------------------

    print("\nSelecting Building 0...")

    X_train_full, y_train_full = prepare_building_data(
        train_df,
        active_waps,
        BUILDING_ID
    )

    X_validation, y_validation = prepare_building_data(
        validation_df,
        active_waps,
        BUILDING_ID
    )

    print(f"Training samples: {X_train_full.shape[0]}")
    print(f"Validation samples: {X_validation.shape[0]}")
    print(f"Features: {X_train_full.shape[1]}")

    # --------------------------------------------------------
    # CREATE INTERNAL VALIDATION SPLIT
    # --------------------------------------------------------
    #
    # We need a validation subset from the TRAINING FILE
    # to choose the best number of boosting iterations.
    #
    # The separate validationData.csv remains untouched
    # until final evaluation.
    # --------------------------------------------------------

    print("\nCreating internal validation split...")

    rng = np.random.default_rng(RANDOM_STATE)

    indices = np.arange(len(X_train_full))
    rng.shuffle(indices)

    split_index = int(len(indices) * 0.8)

    train_indices = indices[:split_index]
    internal_val_indices = indices[split_index:]

    X_train = X_train_full.iloc[train_indices]
    y_train = y_train_full.iloc[train_indices]

    X_internal_val = X_train_full.iloc[internal_val_indices]
    y_internal_val = y_train_full.iloc[internal_val_indices]

    print(f"Model training samples: {len(X_train)}")
    print(f"Internal validation samples: {len(X_internal_val)}")

    # ========================================================
    # 1. TRAIN FULL 300-TREE MODEL
    # ========================================================

    print("\n" + "=" * 60)
    print("TRAINING GRADIENT BOOSTING")
    print("=" * 60)

    print("\nModel parameters:")
    print(f"  Estimators  : {N_ESTIMATORS}")
    print(f"  Max depth   : {MAX_DEPTH}")
    print(f"  Learning rate: {LEARNING_RATE}")

    model = train_gradient_boosting(
        X_train,
        y_train
    )

    print("\nGradient Boosting training complete.")

    # --------------------------------------------------------
    # FIND BEST NUMBER OF TREES
    # --------------------------------------------------------

    print("\nFinding best number of boosting iterations...")

    (
        best_n,
        best_internal_accuracy,
        best_internal_error,
        errors,
        accuracies
    ) = find_best_iterations(
        model,
        X_internal_val,
        y_internal_val
    )

    print(f"\nBest number of trees: {best_n}")
    print(f"Internal validation accuracy: {best_internal_accuracy:.4f}")
    print(f"Internal validation error: {best_internal_error:.4f}")

    # --------------------------------------------------------
    # SAVE ITERATION RESULTS
    # --------------------------------------------------------

    iteration_df = pd.DataFrame({
        "n_estimators": np.arange(1, N_ESTIMATORS + 1),
        "accuracy": accuracies,
        "error": errors
    })

    iteration_path = TABLES_DIR / "gradient_boosting_iterations.csv"

    iteration_df.to_csv(
        iteration_path,
        index=False
    )

    # --------------------------------------------------------
    # PLOT VALIDATION ERROR
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.plot(
        np.arange(1, N_ESTIMATORS + 1),
        errors
    )

    plt.axvline(
        best_n,
        linestyle="--",
        label=f"Best = {best_n}"
    )

    plt.xlabel("Number of Boosting Trees")
    plt.ylabel("Validation Error")
    plt.title("Gradient Boosting Validation Error")
    plt.legend()
    plt.grid(True)

    iteration_plot = (
        FIGURES_DIR /
        "gradient_boosting_iterations.png"
    )

    plt.savefig(
        iteration_plot,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # 2. FINAL MODEL USING BEST NUMBER OF TREES
    # ========================================================

    print("\n" + "=" * 60)
    print("TRAINING FINAL GRADIENT BOOSTING MODEL")
    print("=" * 60)

    final_model = GradientBoostingClassifier(
        n_estimators=best_n,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        random_state=RANDOM_STATE
    )

    print(f"\nTraining samples: {len(X_train_full)}")
    print(f"Best trees: {best_n}")

    final_model.fit(
        X_train_full,
        y_train_full
    )

    print("Final model trained.")

    # --------------------------------------------------------
    # FINAL VALIDATION DATASET EVALUATION
    # --------------------------------------------------------

    print("\nEvaluating on validationData.csv...")

    (
        final_accuracy,
        final_error,
        final_macro_f1
    ) = evaluate_model(
        final_model,
        X_validation,
        y_validation
    )

    print(f"Accuracy : {final_accuracy:.4f}")
    print(f"Error    : {final_error:.4f}")
    print(f"Macro-F1 : {final_macro_f1:.4f}")

    # ========================================================
    # 3. FEATURE IMPORTANCE
    # ========================================================

    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE")
    print("=" * 60)

    importances = final_model.feature_importances_

    feature_importance_df = pd.DataFrame({
        "WAP": active_waps,
        "importance": importances
    })

    feature_importance_df = (
        feature_importance_df
        .sort_values(
            "importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    print("\nTop 20 WAP features:")

    print(
        feature_importance_df.head(20).to_string(
            index=False
        )
    )

    importance_path = (
        TABLES_DIR /
        "gradient_boosting_feature_importance.csv"
    )

    feature_importance_df.to_csv(
        importance_path,
        index=False
    )

    # --------------------------------------------------------
    # TOP 20 FEATURE PLOT
    # --------------------------------------------------------

    top_features = feature_importance_df.head(20)

    plt.figure(figsize=(10, 7))

    plt.barh(
        top_features["WAP"][::-1],
        top_features["importance"][::-1]
    )

    plt.xlabel("Feature Importance")
    plt.ylabel("WAP")
    plt.title("Top 20 WiFi Access Points - Gradient Boosting")

    plt.grid(
        axis="x",
        alpha=0.3
    )

    importance_plot = (
        FIGURES_DIR /
        "gradient_boosting_feature_importance.png"
    )

    plt.savefig(
        importance_plot,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # 4. PARTIAL DEPENDENCE
    # ========================================================

    print("\n" + "=" * 60)
    print("PARTIAL DEPENDENCE")
    print("=" * 60)

    most_important_wap = (
        feature_importance_df.iloc[0]["WAP"]
    )

    most_important_index = active_waps.index(
        most_important_wap
    )

    print(
        f"\nMost important WAP: "
        f"{most_important_wap}"
    )

    print("Generating partial dependence plot...")

    try:

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        PartialDependenceDisplay.from_estimator(
            final_model,
            X_validation,
            features=[most_important_index],
            ax=ax
        )

        ax.set_title(
            f"Partial Dependence - {most_important_wap}"
        )

        pd_plot = (
            FIGURES_DIR /
            "gradient_boosting_partial_dependence.png"
        )

        plt.savefig(
            pd_plot,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print("Partial dependence plot saved.")

    except Exception as e:

        print(
            "\nWARNING: Partial dependence plot "
            "could not be generated."
        )

        print(f"Reason: {e}")

        pd_plot = None

    # ========================================================
    # 5. PER-FLOOR ERROR
    # ========================================================

    print("\n" + "=" * 60)
    print("PER-FLOOR ERROR")
    print("=" * 60)

    predictions = final_model.predict(
        X_validation
    )

    floor_results = []

    for floor in sorted(
        y_validation.unique()
    ):

        mask = (
            y_validation.values == floor
        )

        floor_total = int(mask.sum())

        floor_correct = int(
            (
                predictions[mask]
                == y_validation.values[mask]
            ).sum()
        )

        floor_accuracy = (
            floor_correct / floor_total
            if floor_total > 0
            else 0
        )

        floor_error = 1.0 - floor_accuracy

        floor_results.append({
            "floor": floor,
            "samples": floor_total,
            "accuracy": floor_accuracy,
            "error": floor_error
        })

    floor_df = pd.DataFrame(
        floor_results
    )

    print(
        floor_df.to_string(
            index=False
        )
    )

    floor_path = (
        TABLES_DIR /
        "gradient_boosting_per_floor.csv"
    )

    floor_df.to_csv(
        floor_path,
        index=False
    )

    # --------------------------------------------------------
    # PER-FLOOR ERROR PLOT
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.bar(
        floor_df["floor"].astype(str),
        floor_df["error"]
    )

    plt.xlabel("Floor")
    plt.ylabel("Error")
    plt.title("Gradient Boosting Error by Floor")

    plt.grid(
        axis="y",
        alpha=0.3
    )

    floor_plot = (
        FIGURES_DIR /
        "gradient_boosting_per_floor.png"
    )

    plt.savefig(
        floor_plot,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # 6. SMALL-DATA EXPERIMENT
    # ========================================================

    print("\n" + "=" * 60)
    print("SMALL-DATA EXPERIMENT")
    print("=" * 60)

    sample_results = []

    for sample_size in SAMPLE_SIZES:

        print(
            f"\nTraining with {sample_size} samples..."
        )

        # Use a reproducible random sample.
        sample_size_actual = min(
            sample_size,
            len(X_train_full)
        )

        sample_indices = rng.choice(
            len(X_train_full),
            size=sample_size_actual,
            replace=False
        )

        X_sample = X_train_full.iloc[
            sample_indices
        ]

        y_sample = y_train_full.iloc[
            sample_indices
        ]

        # Use the already discovered best number
        # of trees so this experiment remains fair.
        small_model = GradientBoostingClassifier(
            n_estimators=best_n,
            max_depth=MAX_DEPTH,
            learning_rate=LEARNING_RATE,
            random_state=RANDOM_STATE
        )

        small_model.fit(
            X_sample,
            y_sample
        )

        (
            sample_accuracy,
            sample_error,
            sample_macro_f1
        ) = evaluate_model(
            small_model,
            X_validation,
            y_validation
        )

        print(
            f"Accuracy : {sample_accuracy:.4f}"
        )

        print(
            f"Error    : {sample_error:.4f}"
        )

        print(
            f"Macro-F1 : {sample_macro_f1:.4f}"
        )

        sample_results.append({
            "training_samples": sample_size_actual,
            "accuracy": sample_accuracy,
            "error": sample_error,
            "macro_f1": sample_macro_f1
        })

    sample_df = pd.DataFrame(
        sample_results
    )

    sample_path = (
        TABLES_DIR /
        "gradient_boosting_sample_size.csv"
    )

    sample_df.to_csv(
        sample_path,
        index=False
    )

    # ========================================================
    # 7. FINAL SUMMARY
    # ========================================================

    summary_df = pd.DataFrame([{
        "building": BUILDING_ID,
        "active_wap_features": len(active_waps),
        "best_n_estimators": best_n,
        "max_depth": MAX_DEPTH,
        "learning_rate": LEARNING_RATE,
        "accuracy": final_accuracy,
        "error": final_error,
        "macro_f1": final_macro_f1,
        "top_wap": most_important_wap
    }])

    summary_path = (
        TABLES_DIR /
        "gradient_boosting_results.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    # ========================================================
    # FILE OUTPUT
    # ========================================================

    print("\n" + "=" * 60)
    print("FILES SAVED")
    print("=" * 60)

    print(
        f"\nMain results:\n"
        f"{summary_path}"
    )

    print(
        f"\nIteration results:\n"
        f"{iteration_path}"
    )

    print(
        f"\nFeature importance:\n"
        f"{importance_path}"
    )

    print(
        f"\nPer-floor results:\n"
        f"{floor_path}"
    )

    print(
        f"\nSmall-data results:\n"
        f"{sample_path}"
    )

    print(
        f"\nIteration plot:\n"
        f"{iteration_plot}"
    )

    print(
        f"\nFeature importance plot:\n"
        f"{importance_plot}"
    )

    if pd_plot is not None:
        print(
            f"\nPartial dependence plot:\n"
            f"{pd_plot}"
        )

    print(
        f"\nPer-floor plot:\n"
        f"{floor_plot}"
    )

    print("\n" + "=" * 60)
    print("GRADIENT BOOSTING EXPERIMENT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()