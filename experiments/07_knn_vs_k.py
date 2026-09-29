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
    compute_eigenpairs,
    project_data,
)


ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data" / "trainingData.csv"
VALIDATION_PATH = ROOT / "data" / "validationData.csv"


def evaluate_knn(X, y, k):

    cv = RepeatedStratifiedKFold(
        n_splits=10,
        n_repeats=5,
        random_state=42
    )

    accuracies = []
    macro_f1_scores = []

    for train_idx, test_idx in cv.split(X, y):

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        # -----------------------------
        # Standardization
        # -----------------------------

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(
            X_train
        )

        X_test_scaled = scaler.transform(
            X_test
        )

        # -----------------------------
        # PCA
        # -----------------------------

        covariance_matrix = (
            X_train_scaled.T @ X_train_scaled
        ) / (X_train_scaled.shape[0] - 1)

        eigenvalues, eigenvectors = (
            compute_eigenpairs(
                covariance_matrix
            )
        )

        X_train_pca = project_data(
            X_train_scaled,
            eigenvectors,
            50
        )

        X_test_pca = project_data(
            X_test_scaled,
            eigenvectors,
            50
        )

        # -----------------------------
        # kNN
        # -----------------------------

        model = KNeighborsClassifier(
            n_neighbors=k,
            metric="euclidean"
        )

        model.fit(
            X_train_pca,
            y_train
        )

        y_pred = model.predict(
            X_test_pca
        )

        accuracies.append(
            accuracy_score(
                y_test,
                y_pred
            )
        )

        macro_f1_scores.append(
            f1_score(
                y_test,
                y_pred,
                average="macro"
            )
        )

    return (
        np.mean(accuracies),
        np.std(accuracies),
        np.mean(macro_f1_scores),
        np.std(macro_f1_scores)
    )


def main():

    print("=" * 60)
    print("kNN ERROR VS K")
    print("=" * 60)

    # -----------------------------
    # Load data
    # -----------------------------

    train_df, _ = load_data(
        TRAIN_PATH,
        VALIDATION_PATH
    )

    active_waps = get_active_wap_columns(
        train_df
    )

    X, y = split_by_building(
        train_df,
        active_waps,
        building_id=0
    )

    X = X.values
    y = y.values

    print(
        f"\nDataset shape: {X.shape}"
    )

    # -----------------------------
    # Test k = 1 to 10
    # -----------------------------

    results = []

    for k in range(1, 11):

        print(
            f"\nTesting k = {k}..."
        )

        accuracy_mean, accuracy_std, \
        f1_mean, f1_std = evaluate_knn(
            X,
            y,
            k
        )

        results.append(
            (
                k,
                accuracy_mean,
                accuracy_std,
                f1_mean,
                f1_std
            )
        )

        print(
            f"Accuracy: "
            f"{accuracy_mean:.4f} "
            f"+/- {accuracy_std:.4f}"
        )

        print(
            f"Macro-F1: "
            f"{f1_mean:.4f} "
            f"+/- {f1_std:.4f}"
        )

    # -----------------------------
    # Final comparison
    # -----------------------------

    print("\n" + "=" * 60)
    print("FINAL kNN COMPARISON")
    print("=" * 60)

    print(
        "\nk | Accuracy | Acc Std | Macro-F1 | F1 Std"
    )

    print("-" * 60)

    for row in results:

        k, acc, acc_std, f1, f1_std = row

        print(
            f"{k:2d} | "
            f"{acc:.4f} | "
            f"{acc_std:.4f} | "
            f"{f1:.4f} | "
            f"{f1_std:.4f}"
        )

    # -----------------------------
    # Best k
    # -----------------------------

    best = max(
        results,
        key=lambda x: x[1]
    )

    print("\n" + "=" * 60)

    print(
        f"BEST k = {best[0]}"
    )

    print(
        f"Accuracy = {best[1]:.4f}"
    )

    print(
        f"Macro-F1 = {best[3]:.4f}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()