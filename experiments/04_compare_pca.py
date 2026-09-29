from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

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
    project_data,
    reconstruct_data,
)


ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data" / "trainingData.csv"
VALIDATION_PATH = ROOT / "data" / "validationData.csv"


def main():

    print("=" * 60)
    print("PCA: FROM SCRATCH vs SKLEARN")
    print("=" * 60)

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------

    train_df, _ = load_data(
        TRAIN_PATH,
        VALIDATION_PATH
    )

    active_waps = get_active_wap_columns(train_df)

    X, _ = split_by_building(
        train_df,
        active_waps,
        building_id=0
    )

    # --------------------------------------------------
    # Standardize
    # --------------------------------------------------

    X_standardized, mean, std = standardize(
        X.values
    )

    # ==================================================
    # PCA FROM SCRATCH
    # ==================================================

    covariance_matrix = compute_covariance_matrix(
        X_standardized
    )

    eigenvalues, eigenvectors = compute_eigenpairs(
        covariance_matrix
    )

    scratch_variance = explained_variance_ratio(
        eigenvalues
    )

    # ==================================================
    # SKLEARN PCA
    # ==================================================

    sklearn_pca = PCA()

    sklearn_pca.fit(X_standardized)

    sklearn_variance = (
        sklearn_pca.explained_variance_ratio_
    )

    # --------------------------------------------------
    # Compare explained variance
    # --------------------------------------------------

    print("\n--- Explained Variance Comparison ---")

    print(
        f"First component - Scratch: "
        f"{scratch_variance[0]:.10f}"
    )

    print(
        f"First component - sklearn: "
        f"{sklearn_variance[0]:.10f}"
    )

    print(
        f"\nFirst 10 components - Scratch:"
    )

    print(scratch_variance[:10])

    print(
        f"\nFirst 10 components - sklearn:"
    )

    print(sklearn_variance[:10])

    # --------------------------------------------------
    # Numerical difference
    # --------------------------------------------------

    variance_difference = np.abs(
        scratch_variance - sklearn_variance
    )

    print(
        "\nMaximum explained-variance difference:"
    )

    print(
        f"{variance_difference.max():.12f}"
    )

    # --------------------------------------------------
    # Compare cumulative variance
    # --------------------------------------------------

    scratch_cumulative = np.cumsum(
        scratch_variance
    )

    sklearn_cumulative = np.cumsum(
        sklearn_variance
    )

    print("\n--- Components Required ---")

    for threshold in [0.90, 0.95, 0.99]:

        scratch_k = (
            np.argmax(
                scratch_cumulative >= threshold
            ) + 1
        )

        sklearn_k = (
            np.argmax(
                sklearn_cumulative >= threshold
            ) + 1
        )

        print(
            f"{threshold * 100:.0f}% variance:"
        )

        print(
            f"  Scratch : {scratch_k}"
        )

        print(
            f"  sklearn : {sklearn_k}"
        )

    # ==================================================
    # Reconstruction error
    # ==================================================

    print("\n--- Reconstruction Error ---")

    for k in [20, 50, 100, 109, 133, 167]:

        # ------------------------------
        # Scratch PCA
        # ------------------------------

        X_projected = project_data(
            X_standardized,
            eigenvectors,
            k
        )

        X_reconstructed = reconstruct_data(
            X_projected,
            eigenvectors,
            k
        )

        scratch_mse = np.mean(
            (X_standardized - X_reconstructed) ** 2
        )

        # ------------------------------
        # sklearn PCA
        # ------------------------------

        sklearn_model = PCA(
            n_components=k
        )

        X_sklearn_projected = (
            sklearn_model.fit_transform(
                X_standardized
            )
        )

        X_sklearn_reconstructed = (
            sklearn_model.inverse_transform(
                X_sklearn_projected
            )
        )

        sklearn_mse = np.mean(
            (
                X_standardized
                - X_sklearn_reconstructed
            ) ** 2
        )

        print(
            f"{k:3d} components:"
            f" Scratch MSE = {scratch_mse:.8f}"
            f" | sklearn MSE = {sklearn_mse:.8f}"
        )

    print("\nPCA comparison complete.")


if __name__ == "__main__":
    main()