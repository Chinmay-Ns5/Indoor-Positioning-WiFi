"""
SVM EXPERIMENTS

Experiments:
1. Compare SVM performance for different PCA dimensions.
2. Study effect of training sample size.
3. Save results to CSV.

Uses PCA implemented from scratch.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

from src.data_utils import (
    load_data,
    get_active_wap_columns,
    split_by_building,
)

from src.pca_from_scratch import (
    standardize,
    compute_covariance_matrix,
    compute_eigenpairs,
    project_data,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data" / "trainingData.csv"
VALIDATION_PATH = ROOT / "data" / "validationData.csv"

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

BUILDING_ID = 0

# PCA dimensions to test
PCA_COMPONENTS = [50, 100, 200]

# Training sample sizes to test
SAMPLE_SIZES = [100, 1000, 2000]

# Fixed number of samples for PCA comparison
FIXED_SAMPLE_SIZE = 2000

# SVM settings
SVM_C = 10
SVM_KERNEL = "rbf"

# Reproducibility
RANDOM_STATE = 42


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def transform_with_pca(
    X_train,
    X_test,
    n_components
):
    """
    Standardize training data, apply the same transformation
    to test data, then perform PCA using the from-scratch
    implementation.

    Returns:
        X_train_pca
        X_test_pca
    """

    # --------------------------------------------------------
    # Standardize training data
    # --------------------------------------------------------

    X_train_scaled, mean, std = standardize(X_train)

    # --------------------------------------------------------
    # Apply SAME training transformation to test data
    # --------------------------------------------------------

    X_test = np.asarray(X_test, dtype=float)

    X_test_scaled = (X_test - mean) / std

    # --------------------------------------------------------
    # Compute covariance matrix
    # --------------------------------------------------------

    covariance_matrix = compute_covariance_matrix(
        X_train_scaled
    )

    # --------------------------------------------------------
    # Eigen decomposition
    # --------------------------------------------------------

    eigenvalues, eigenvectors = compute_eigenpairs(
        covariance_matrix
    )

    # --------------------------------------------------------
    # PCA projection
    # --------------------------------------------------------

    X_train_pca = project_data(
        X_train_scaled,
        eigenvectors,
        n_components
    )

    X_test_pca = project_data(
        X_test_scaled,
        eigenvectors,
        n_components
    )

    return X_train_pca, X_test_pca


def evaluate_svm(
    X_train,
    y_train,
    X_test,
    y_test,
    n_components
):
    """
    Apply PCA and train/evaluate an SVM.
    """

    # --------------------------------------------------------
    # PCA
    # --------------------------------------------------------

    X_train_pca, X_test_pca = transform_with_pca(
        X_train,
        X_test,
        n_components
    )

    # --------------------------------------------------------
    # Train SVM
    # --------------------------------------------------------

    model = SVC(
        kernel=SVM_KERNEL,
        C=SVM_C,
        gamma="scale"
    )

    model.fit(
        X_train_pca,
        y_train
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predictions = model.predict(X_test_pca)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro"
    )

    error = 1.0 - accuracy

    return accuracy, error, macro_f1


# ============================================================
# PCA COMPONENT EXPERIMENT
# ============================================================

def run_pca_experiment(
    X_train,
    y_train,
    X_test,
    y_test
):
    """
    Compare SVM performance for different PCA dimensions.
    """

    print()
    print("=" * 60)
    print("SVM PCA COMPONENT EXPERIMENT")
    print("=" * 60)

    results = []

    # --------------------------------------------------------
    # Use fixed number of training samples
    # --------------------------------------------------------

    sample_size = min(
        FIXED_SAMPLE_SIZE,
        X_train.shape[0]
    )

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    indices = rng.choice(
        X_train.shape[0],
        size=sample_size,
        replace=False
    )

    X_sample = X_train[indices]
    y_sample = y_train[indices]

    print()
    print(f"Training samples used: {sample_size}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Original features: {X_train.shape[1]}")

    # --------------------------------------------------------
    # Test PCA dimensions
    # --------------------------------------------------------

    for n_components in PCA_COMPONENTS:

        # PCA cannot have more components than:
        # min(number of samples, number of features)

        max_components = min(
            X_sample.shape[0],
            X_sample.shape[1]
        )

        if n_components > max_components:

            print()
            print(
                f"Skipping PCA={n_components}: "
                f"maximum allowed is {max_components}"
            )

            continue

        print()
        print("-" * 50)
        print(f"PCA = {n_components}")
        print("-" * 50)

        print("Training SVM...")

        accuracy, error, macro_f1 = evaluate_svm(
            X_sample,
            y_sample,
            X_test,
            y_test,
            n_components
        )

        print(
            f"Accuracy: {accuracy:.4f}"
        )

        print(
            f"Error: {error:.4f}"
        )

        print(
            f"Macro-F1: {macro_f1:.4f}"
        )

        results.append({
            "PCA Components": n_components,
            "Training Samples": sample_size,
            "Accuracy": accuracy,
            "Error": error,
            "Macro-F1": macro_f1
        })

    return pd.DataFrame(results)


# ============================================================
# SAMPLE SIZE EXPERIMENT
# ============================================================

def run_sample_size_experiment(
    X_train,
    y_train,
    X_test,
    y_test
):
    """
    Study how SVM performance changes with training size.

    PCA is kept at 50 components so that even the smallest
    sample size can be evaluated safely.
    """

    print()
    print("=" * 60)
    print("SVM SAMPLE SIZE EXPERIMENT")
    print("=" * 60)

    results = []

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    for sample_size in SAMPLE_SIZES:

        # ----------------------------------------------------
        # Make sure requested size exists
        # ----------------------------------------------------

        if sample_size > X_train.shape[0]:

            print()
            print(
                f"Skipping sample size {sample_size}: "
                f"only {X_train.shape[0]} training samples available."
            )

            continue

        # ----------------------------------------------------
        # PCA dimension
        # ----------------------------------------------------

        n_components = 50

        # 50 components must be valid
        max_components = min(
            sample_size,
            X_train.shape[1]
        )

        if n_components > max_components:

            print()
            print(
                f"Skipping sample size {sample_size}: "
                f"PCA={n_components} is invalid."
            )

            continue

        # ----------------------------------------------------
        # Random subset
        # ----------------------------------------------------

        indices = rng.choice(
            X_train.shape[0],
            size=sample_size,
            replace=False
        )

        X_sample = X_train[indices]
        y_sample = y_train[indices]

        print()
        print("-" * 50)
        print(f"Training SVM:")
        print(f"  Samples = {sample_size}")
        print(f"  PCA     = {n_components}")
        print("-" * 50)

        accuracy, error, macro_f1 = evaluate_svm(
            X_sample,
            y_sample,
            X_test,
            y_test,
            n_components
        )

        print(
            f"Accuracy: {accuracy:.4f}"
        )

        print(
            f"Error: {error:.4f}"
        )

        print(
            f"Macro-F1: {macro_f1:.4f}"
        )

        results.append({
            "Training Samples": sample_size,
            "PCA Components": n_components,
            "Accuracy": accuracy,
            "Error": error,
            "Macro-F1": macro_f1
        })

    return pd.DataFrame(results)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("SVM EXPERIMENTS")
    print("=" * 60)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print()
    print("Loading data...")

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
    # Select active WAPs
    # --------------------------------------------------------

    print()
    print("Selecting active WAPs...")

    active_waps = get_active_wap_columns(
        train_df
    )

    print(
        f"Active WAP features: {len(active_waps)}"
    )

    # --------------------------------------------------------
    # Extract Building 0
    # --------------------------------------------------------

    print()
    print("Extracting Building 0...")

    X_building, y_building = split_by_building(
        train_df,
        active_waps,
        BUILDING_ID
    )

    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    X_building = np.asarray(
        X_building,
        dtype=float
    )

    y_building = np.asarray(
        y_building
    ).ravel()

    print(
        f"Building 0 samples: {X_building.shape[0]}"
    )

    print(
        f"WAP features: {X_building.shape[1]}"
    )

    # --------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X_building,
        y_building,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_building
    )

    print()
    print("Train/test split:")
    print(
        f"Training samples: {X_train.shape[0]}"
    )
    print(
        f"Test samples: {X_test.shape[0]}"
    )

    # ========================================================
    # EXPERIMENT 1
    # ========================================================

    pca_results = run_pca_experiment(
        X_train,
        y_train,
        X_test,
        y_test
    )

    # --------------------------------------------------------
    # Save PCA results
    # --------------------------------------------------------

    pca_output = (
        RESULTS_DIR /
        "svm_pca_results.csv"
    )

    pca_results.to_csv(
        pca_output,
        index=False
    )

    # ========================================================
    # EXPERIMENT 2
    # ========================================================

    sample_results = run_sample_size_experiment(
        X_train,
        y_train,
        X_test,
        y_test
    )

    # --------------------------------------------------------
    # Save sample-size results
    # --------------------------------------------------------

    sample_output = (
        RESULTS_DIR /
        "svm_sample_size_results.csv"
    )

    sample_results.to_csv(
        sample_output,
        index=False
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 60)
    print("FINAL SVM RESULTS")
    print("=" * 60)

    if not pca_results.empty:

        print()
        print("PCA RESULTS")
        print("-" * 60)

        print(
            pca_results.to_string(
                index=False
            )
        )

        best_accuracy_row = pca_results.loc[
            pca_results["Accuracy"].idxmax()
        ]

        best_f1_row = pca_results.loc[
            pca_results["Macro-F1"].idxmax()
        ]

        print()
        print(
            f"Best PCA by Accuracy: "
            f"{int(best_accuracy_row['PCA Components'])} "
            f"components"
        )

        print(
            f"Accuracy: "
            f"{best_accuracy_row['Accuracy']:.4f}"
        )

        print()
        print(
            f"Best PCA by Macro-F1: "
            f"{int(best_f1_row['PCA Components'])} "
            f"components"
        )

        print(
            f"Macro-F1: "
            f"{best_f1_row['Macro-F1']:.4f}"
        )

    if not sample_results.empty:

        print()
        print("SAMPLE SIZE RESULTS")
        print("-" * 60)

        print(
            sample_results.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Output locations
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("FILES SAVED")
    print("=" * 60)

    print(
        f"PCA results: {pca_output}"
    )

    print(
        f"Sample-size results: {sample_output}"
    )

    print()
    print("=" * 60)
    print("SVM EXPERIMENTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()