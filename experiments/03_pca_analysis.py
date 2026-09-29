from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data_utils import (
    load_data,
    get_active_wap_columns,
    split_by_building,
)

from src.pca_from_scratch import (
    standardize,
    compute_covariance_matrix,
    compute_eigenpairs,
    explained_variance_ratio,
    cumulative_explained_variance,
)


ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data" / "trainingData.csv"
VALIDATION_PATH = ROOT / "data" / "validationData.csv"

FIGURES_DIR = ROOT / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def main():

    print("=" * 60)
    print("PCA FROM SCRATCH")
    print("=" * 60)

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------

    train_df, _ = load_data(
        TRAIN_PATH,
        VALIDATION_PATH
    )

    # --------------------------------------------------
    # Select active WAPs
    # --------------------------------------------------

    active_waps = get_active_wap_columns(train_df)

    # --------------------------------------------------
    # Use Building 0 for initial PCA analysis
    # --------------------------------------------------

    X, y = split_by_building(
        train_df,
        active_waps,
        building_id=0
    )

    print(f"\nBuilding 0 data: {X.shape}")

    # --------------------------------------------------
    # Standardize
    # --------------------------------------------------

    X_standardized, mean, std = standardize(X.values)

    print(
        f"Standardized data shape: "
        f"{X_standardized.shape}"
    )

    # --------------------------------------------------
    # Covariance matrix
    # --------------------------------------------------

    covariance_matrix = compute_covariance_matrix(
        X_standardized
    )

    print(
        f"Covariance matrix shape: "
        f"{covariance_matrix.shape}"
    )

    # --------------------------------------------------
    # Eigen decomposition
    # --------------------------------------------------

    eigenvalues, eigenvectors = compute_eigenpairs(
        covariance_matrix
    )

    print(
        f"Number of eigenvalues: "
        f"{len(eigenvalues)}"
    )

    # --------------------------------------------------
    # Explained variance
    # --------------------------------------------------

    variance_ratio = explained_variance_ratio(
        eigenvalues
    )

    cumulative_variance = cumulative_explained_variance(
        eigenvalues
    )

    print("\nFirst 10 eigenvalues:")
    print(eigenvalues[:10])

    print("\nFirst 10 explained variance ratios:")
    print(variance_ratio[:10])

    print("\nCumulative explained variance:")

    for k in [2, 5, 10, 20, 30, 50, 100]:
        if k <= len(cumulative_variance):
            print(
                f"{k:3d} components: "
                f"{cumulative_variance[k - 1] * 100:.2f}%"
            )

    # --------------------------------------------------
    # Find components needed for 90%, 95%, 99%
    # --------------------------------------------------

    thresholds = [0.90, 0.95, 0.99]

    print("\nComponents required:")

    for threshold in thresholds:

        k = (
            cumulative_variance >= threshold
        ).argmax() + 1

        print(
            f"{threshold * 100:.0f}% variance: "
            f"{k} components"
        )

    # --------------------------------------------------
    # Plot cumulative explained variance
    # --------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        range(1, len(cumulative_variance) + 1),
        cumulative_variance
    )

    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Explained Variance")

    plt.grid(True)

    output_path = (
        FIGURES_DIR /
        "pca_explained_variance.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\nPlot saved to:\n{output_path}"
    )

    print("\nPCA analysis complete.")


if __name__ == "__main__":
    main()