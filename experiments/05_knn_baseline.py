from pathlib import Path

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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


ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data" / "trainingData.csv"
VALIDATION_PATH = ROOT / "data" / "validationData.csv"


def main():

    print("=" * 60)
    print("kNN BASELINE - BUILDING 0")
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

    print(
        f"\nBuilding 0 dataset: "
        f"{X.shape}"
    )

    # --------------------------------------------------
    # Train/test split
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
    )

    print(
        f"Training samples: "
        f"{len(X_train)}"
    )

    print(
        f"Test samples: "
        f"{len(X_test)}"
    )

    # --------------------------------------------------
    # Standardization
    # --------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # --------------------------------------------------
    # PCA
    # --------------------------------------------------

    covariance_matrix = (
        X_train_scaled.T @ X_train_scaled
    ) / (X_train_scaled.shape[0] - 1)

    eigenvalues, eigenvectors = (
        compute_eigenpairs(
            covariance_matrix
        )
    )

    # Try 50 PCA components
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

    print(
        f"\nPCA components: "
        f"{n_components}"
    )

    print(
        f"Reduced shape: "
        f"{X_train_pca.shape}"
    )

    # --------------------------------------------------
    # kNN
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro"
    )

    print("\n--- Results ---")

    print(
        f"Accuracy: "
        f"{accuracy:.4f}"
    )

    print(
        f"Macro-F1: "
        f"{macro_f1:.4f}"
    )

    print("\nkNN baseline complete.")


if __name__ == "__main__":
    main()