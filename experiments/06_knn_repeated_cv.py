from pathlib import Path

import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

from src.data_utils import (
    load_data,
    get_active_wap_columns,
    split_by_building,
)

from src.pca_from_scratch import (
    compute_covariance_matrix,
    compute_eigenpairs,
    project_data,
)


ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data" / "trainingData.csv"
VALIDATION_PATH = ROOT / "data" / "validationData.csv"


def main():

    print("=" * 60)
    print("kNN - REPEATED 10-FOLD CROSS-VALIDATION")
    print("=" * 60)

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------

    train_df, _ = load_data(
        TRAIN_PATH,
        VALIDATION_PATH
    )

    active_waps = get_active_wap_columns(
        train_df
    )

    # --------------------------------------------------
    # Building 0
    # --------------------------------------------------

    X, y = split_by_building(
        train_df,
        active_waps,
        building_id=0
    )

    X = X.values
    y = y.values

    print(
        f"\nDataset: {X.shape}"
    )

    # --------------------------------------------------
    # CV configuration
    # --------------------------------------------------

    cv = RepeatedStratifiedKFold(
        n_splits=10,
        n_repeats=5,
        random_state=42
    )

    accuracies = []
    macro_f1_scores = []

    # --------------------------------------------------
    # Run CV
    # --------------------------------------------------

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y),
        start=1
    ):

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        # ------------------------------
        # Standardization
        # ------------------------------

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(
            X_train
        )

        X_test_scaled = scaler.transform(
            X_test
        )

        # ------------------------------
        # PCA
        # ------------------------------

        covariance_matrix = (
            X_train_scaled.T @ X_train_scaled
        ) / (X_train_scaled.shape[0] - 1)

        eigenvalues, eigenvectors = (
            compute_eigenpairs(
                covariance_matrix
            )
        )

        n_components = 50

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

        # ------------------------------
        # kNN
        # ------------------------------

        model = KNeighborsClassifier(
            n_neighbors=5,
            metric="euclidean"
        )

        model.fit(
            X_train_pca,
            y_train
        )

        y_pred = model.predict(
            X_test_pca
        )

        # ------------------------------
        # Metrics
        # ------------------------------

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        macro_f1 = f1_score(
            y_test,
            y_pred,
            average="macro"
        )

        accuracies.append(
            accuracy
        )

        macro_f1_scores.append(
            macro_f1
        )

        print(
            f"Fold {fold:02d}/50 | "
            f"Accuracy = {accuracy:.4f} | "
            f"Macro-F1 = {macro_f1:.4f}"
        )

    # --------------------------------------------------
    # Final results
    # --------------------------------------------------

    accuracies = np.array(
        accuracies
    )

    macro_f1_scores = np.array(
        macro_f1_scores
    )

    print("\n" + "=" * 60)
    print("FINAL 10-FOLD × 5-REPEAT RESULTS")
    print("=" * 60)

    print(
        f"\nAccuracy:"
        f" {accuracies.mean():.4f}"
        f" ± {accuracies.std():.4f}"
    )

    print(
        f"Macro-F1:"
        f" {macro_f1_scores.mean():.4f}"
        f" ± {macro_f1_scores.std():.4f}"
    )

    print(
        "\nRepeated cross-validation complete."
    )


if __name__ == "__main__":
    main()