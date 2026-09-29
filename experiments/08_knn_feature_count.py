import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

from src.data_utils import (
    load_data,
    get_active_wap_columns,
    split_by_building
)

from src.pca_from_scratch import (
    standardize,
    compute_covariance_matrix,
    compute_eigenpairs,
    project_data
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
# PCA + kNN EVALUATION
# ============================================================

def evaluate_knn(
    X_train,
    y_train,
    X_test,
    y_test,
    eigenvectors,
    n_components,
    k=5
):
    """
    Project data using PCA and evaluate kNN.
    """

    # --------------------------------------------------------
    # PCA projection
    # --------------------------------------------------------

    X_train_pca = project_data(
        X_train,
        eigenvectors,
        n_components
    )

    X_test_pca = project_data(
        X_test,
        eigenvectors,
        n_components
    )

    # --------------------------------------------------------
    # kNN
    # --------------------------------------------------------

    knn = KNeighborsClassifier(
        n_neighbors=k
    )

    knn.fit(
        X_train_pca,
        y_train
    )

    y_pred = knn.predict(
        X_test_pca
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro"
    )

    return accuracy, macro_f1


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("kNN FEATURE COUNT EXPERIMENT")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Load data
    # --------------------------------------------------------

    train_df, validation_df = load_data(
        TRAIN_PATH,
        VALIDATION_PATH
    )

    print("\nData loaded.")
    print(f"Training shape:   {train_df.shape}")
    print(f"Validation shape: {validation_df.shape}")

    # --------------------------------------------------------
    # 2. Select active WAPs
    # IMPORTANT:
    # Active WAPs are determined ONLY from training data.
    # --------------------------------------------------------

    active_waps = get_active_wap_columns(
        train_df
    )

    print("\nWAP feature selection:")
    print(f"Total WAP columns:   {len(get_active_wap_columns(train_df))}")
    print(f"Active WAP columns:  {len(active_waps)}")
    print(f"Removed WAP columns: {520 - len(active_waps)}")

    # --------------------------------------------------------
    # 3. Extract Building 0
    # --------------------------------------------------------

    building_id = 0

    X_train, y_train = split_by_building(
        train_df,
        active_waps,
        building_id=building_id
    )

    X_test, y_test = split_by_building(
        validation_df,
        active_waps,
        building_id=building_id
    )

    # Convert to numpy
    X_train = X_train.values
    X_test = X_test.values

    y_train = y_train.values
    y_test = y_test.values

    print("\nBuilding 0:")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples:     {X_test.shape[0]}")
    print(f"Active WAP features: {X_train.shape[1]}")

    # --------------------------------------------------------
    # 4. Standardization
    # --------------------------------------------------------
    # IMPORTANT:
    # Calculate mean/std ONLY from training data.
    # Apply the same transformation to test data.
    # --------------------------------------------------------

    X_train_scaled, mean, std = standardize(
        X_train
    )

    X_test_scaled = (
        X_test - mean
    ) / std

    print("\nStandardization complete.")

    # --------------------------------------------------------
    # 5. PCA from scratch
    # --------------------------------------------------------

    print("\nComputing covariance matrix...")

    covariance_matrix = compute_covariance_matrix(
        X_train_scaled
    )

    print(
        f"Covariance matrix shape: "
        f"{covariance_matrix.shape}"
    )

    print("\nComputing eigenvalues/eigenvectors...")

    eigenvalues, eigenvectors = compute_eigenpairs(
        covariance_matrix
    )

    print(
        f"Number of eigenvalues: "
        f"{len(eigenvalues)}"
    )

    # --------------------------------------------------------
    # 6. PCA feature-count experiment
    # --------------------------------------------------------

    component_values = [
        5,
        50,
        100,
        200
    ]

    results = []

    print("\n" + "=" * 60)
    print("TESTING PCA FEATURE COUNTS")
    print("=" * 60)

    for n_components in component_values:

        print(
            f"\nTesting {n_components} PCA components..."
        )

        # Make sure we don't request more components
        # than available features.
        if n_components > X_train.shape[1]:
            print(
                f"Skipping {n_components}: "
                f"only {X_train.shape[1]} features available."
            )
            continue

        accuracy, macro_f1 = evaluate_knn(
            X_train_scaled,
            y_train,
            X_test_scaled,
            y_test,
            eigenvectors,
            n_components,
            k=5
        )

        print(
            f"Accuracy: {accuracy:.4f}"
        )

        print(
            f"Macro-F1: {macro_f1:.4f}"
        )

        results.append({
            "PCA Components": n_components,
            "Accuracy": accuracy,
            "Macro-F1": macro_f1
        })

    # --------------------------------------------------------
    # 7. Results table
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    print("\n" + "=" * 60)
    print("FINAL FEATURE COUNT RESULTS")
    print("=" * 60)

    print(
        results_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # 8. Find best feature count
    # --------------------------------------------------------

    if not results_df.empty:

        best_accuracy_row = results_df.loc[
            results_df["Accuracy"].idxmax()
        ]

        best_f1_row = results_df.loc[
            results_df["Macro-F1"].idxmax()
        ]

        print("\n" + "=" * 60)
        print("BEST RESULTS")
        print("=" * 60)

        print(
            f"Best Accuracy:"
        )

        print(
            f"  PCA components = "
            f"{int(best_accuracy_row['PCA Components'])}"
        )

        print(
            f"  Accuracy = "
            f"{best_accuracy_row['Accuracy']:.4f}"
        )

        print(
            f"\nBest Macro-F1:"
        )

        print(
            f"  PCA components = "
            f"{int(best_f1_row['PCA Components'])}"
        )

        print(
            f"  Macro-F1 = "
            f"{best_f1_row['Macro-F1']:.4f}"
        )

    # --------------------------------------------------------
    # 9. Save results
    # --------------------------------------------------------

    output_path = (
        RESULTS_DIR /
        "knn_feature_count.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nResults saved to:"
        f"\n{output_path}"
    )

    print("\n" + "=" * 60)
    print("kNN FEATURE COUNT EXPERIMENT COMPLETE")
    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()