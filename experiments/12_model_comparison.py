"""
STEP 10
Big Model Comparison + kNN + Gradient Boosting Combination

Models:
    - kNN
    - SVM
    - Decision Tree
    - Gradient Boosting

Sample sizes:
    100, 200, 500, 1000, 2000, 5000

Combinations:
    1. Switch rule
    2. Probability voting: kNN (k=3) + Gradient Boosting

Detailed experiment:
    Building 0

Final compact comparison:
    Buildings 0, 1, 2
"""

import os
import sys
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ============================================================
# DATA PATHS
# ============================================================

TRAIN_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "trainingData.csv"
)

VALIDATION_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "validationData.csv"
)

# ============================================================
# IMPORT DATA UTILITIES
# ============================================================

from src.data_utils import (
    load_data,
    get_active_wap_columns,
    split_by_building
)


# ============================================================
# CONFIGURATION
# ============================================================

BUILDING_ID = 0

SAMPLE_SIZES = [100, 200, 500, 1000, 2000, 5000]

# SVM becomes expensive with larger sample sizes
SVM_MAX_SAMPLES = 2000

PCA_COMPONENTS = 50

# kNN settings
KNN_K = 1

# Voting kNN must use k=3 to obtain useful probabilities
VOTING_KNN_K = 3

# Gradient Boosting settings
GB_N_ESTIMATORS = 300
GB_MAX_DEPTH = 4
GB_LEARNING_RATE = 0.1
GB_RANDOM_STATE = 0

# Randomness
RANDOM_STATE = 42


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ============================================================
# HELPER: SAMPLE TRAINING DATA
# ============================================================

def sample_training_data(X, y, n_samples, random_state=RANDOM_STATE):
    """
    Randomly select n_samples from the training data.

    If n_samples is larger than the available data,
    use the full training set.
    """

    n_samples = min(n_samples, len(X))

    rng = np.random.RandomState(random_state)

    indices = rng.choice(
        len(X),
        size=n_samples,
        replace=False
    )

    return X[indices], y[indices]


# ============================================================
# HELPER: PREPARE PCA DATA
# ============================================================

def prepare_pca_data(X_train, X_test, n_components=PCA_COMPONENTS):
    """
    Standardize training data and test data using the
    training transformation, then apply PCA.

    PCA is fitted ONLY on training data.
    """

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    # Never request more components than mathematically possible
    max_components = min(
        X_train_scaled.shape[0],
        X_train_scaled.shape[1]
    )

    actual_components = min(
        n_components,
        max_components
    )

    pca = PCA(
        n_components=actual_components,
        random_state=RANDOM_STATE
    )

    X_train_pca = pca.fit_transform(X_train_scaled)

    X_test_pca = pca.transform(X_test_scaled)

    return (
        X_train_pca,
        X_test_pca,
        scaler,
        pca,
        actual_components
    )


# ============================================================
# FROM-SCRATCH kNN
# ============================================================

def knn_predict(X_train, y_train, X_test, k=1):
    """
    Simple from-scratch kNN classifier.

    Uses Euclidean distance.
    """

    predictions = []

    for test_point in X_test:

        distances = np.sqrt(
            np.sum(
                (X_train - test_point) ** 2,
                axis=1
            )
        )

        nearest_indices = np.argsort(distances)[:k]

        nearest_labels = y_train[nearest_indices]

        labels, counts = np.unique(
            nearest_labels,
            return_counts=True
        )

        prediction = labels[np.argmax(counts)]

        predictions.append(prediction)

    return np.array(predictions)


def knn_predict_proba(
    X_train,
    y_train,
    X_test,
    classes,
    k=3
):
    """
    From-scratch kNN probability estimation.

    Probability = fraction of the k nearest neighbours
    belonging to each class.
    """

    probabilities = np.zeros(
        (len(X_test), len(classes)),
        dtype=float
    )

    class_to_index = {
        cls: i
        for i, cls in enumerate(classes)
    }

    for row_index, test_point in enumerate(X_test):

        distances = np.sqrt(
            np.sum(
                (X_train - test_point) ** 2,
                axis=1
            )
        )

        nearest_indices = np.argsort(distances)[:k]

        nearest_labels = y_train[nearest_indices]

        for label in nearest_labels:

            class_index = class_to_index[label]

            probabilities[
                row_index,
                class_index
            ] += 1.0 / k

    return probabilities


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):
    """
    Calculate accuracy, error and macro-F1.
    """

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    error = 1.0 - accuracy

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro"
    )

    return accuracy, error, macro_f1


# ============================================================
# TRAIN + EVALUATE kNN
# ============================================================

def evaluate_knn(
    X_train,
    y_train,
    X_test,
    y_test,
    k=1
):

    y_pred = knn_predict(
        X_train,
        y_train,
        X_test,
        k=k
    )

    return calculate_metrics(
        y_test,
        y_pred
    )


# ============================================================
# TRAIN + EVALUATE SVM
# ============================================================

def evaluate_svm(
    X_train,
    y_train,
    X_test,
    y_test
):

    model = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        random_state=RANDOM_STATE
    )

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(
        X_test
    )

    return calculate_metrics(
        y_test,
        y_pred
    )


# ============================================================
# TRAIN + EVALUATE DECISION TREE
# ============================================================

def evaluate_tree(
    X_train,
    y_train,
    X_test,
    y_test
):

    # Conservative pruning.
    # This prevents an unnecessarily large tree.
    model = DecisionTreeClassifier(
        max_depth=20,
        min_samples_leaf=2,
        random_state=RANDOM_STATE
    )

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(
        X_test
    )

    return calculate_metrics(
        y_test,
        y_pred
    )


# ============================================================
# TRAIN + EVALUATE GRADIENT BOOSTING
# ============================================================

def evaluate_gradient_boosting(
    X_train,
    y_train,
    X_test,
    y_test
):

    model = GradientBoostingClassifier(
        n_estimators=GB_N_ESTIMATORS,
        max_depth=GB_MAX_DEPTH,
        learning_rate=GB_LEARNING_RATE,
        random_state=GB_RANDOM_STATE
    )

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(
        X_test
    )

    return (
        *calculate_metrics(
            y_test,
            y_pred
        ),
        model
    )


# ============================================================
# PROBABILITY VOTING
# ============================================================

def evaluate_probability_voting(
    X_train,
    y_train,
    X_test,
    y_test
):
    """
    Combination 2:

        kNN (k=3) probability
              +
        Gradient Boosting probability

    Average the probabilities and select
    the class with maximum probability.
    """

    classes = np.unique(y_train)

    # --------------------------------------------------------
    # kNN probabilities
    # --------------------------------------------------------

    knn_probabilities = knn_predict_proba(
        X_train,
        y_train,
        X_test,
        classes,
        k=VOTING_KNN_K
    )

    # --------------------------------------------------------
    # Gradient Boosting
    # --------------------------------------------------------

    gb_model = GradientBoostingClassifier(
        n_estimators=GB_N_ESTIMATORS,
        max_depth=GB_MAX_DEPTH,
        learning_rate=GB_LEARNING_RATE,
        random_state=GB_RANDOM_STATE
    )

    gb_model.fit(
        X_train,
        y_train
    )

    gb_probabilities_raw = gb_model.predict_proba(
        X_test
    )

    # Align GB probability columns with classes
    gb_probabilities = np.zeros_like(
        knn_probabilities
    )

    for i, cls in enumerate(gb_model.classes_):

        class_index = np.where(
            classes == cls
        )[0][0]

        gb_probabilities[:, class_index] = (
            gb_probabilities_raw[:, i]
        )

    # --------------------------------------------------------
    # Average probabilities
    # --------------------------------------------------------

    average_probabilities = (
        knn_probabilities +
        gb_probabilities
    ) / 2.0

    prediction_indices = np.argmax(
        average_probabilities,
        axis=1
    )

    y_pred = classes[
        prediction_indices
    ]

    return calculate_metrics(
        y_test,
        y_pred
    )


# ============================================================
# MAIN MODEL COMPARISON
# ============================================================

def run_model_comparison(
    X_train,
    y_train,
    X_test,
    y_test
):

    results = []

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    for n_samples in SAMPLE_SIZES:

        print(
            f"\n{'-' * 60}"
        )

        print(
            f"Training sample size: {n_samples}"
        )

        X_sample, y_sample = sample_training_data(
            X_train,
            y_train,
            n_samples
        )

        # ----------------------------------------------------
        # PCA
        # ----------------------------------------------------

        (
            X_sample_pca,
            X_test_pca,
            scaler,
            pca,
            actual_components
        ) = prepare_pca_data(
            X_sample,
            X_test,
            PCA_COMPONENTS
        )

        print(
            f"PCA components: {actual_components}"
        )

        # ----------------------------------------------------
        # kNN
        # ----------------------------------------------------

        print("Running kNN...")

        accuracy, error, macro_f1 = evaluate_knn(
            X_sample_pca,
            y_sample,
            X_test_pca,
            y_test,
            k=KNN_K
        )

        print(
            f"kNN       Accuracy: {accuracy:.4f} "
            f"Error: {error:.4f} "
            f"Macro-F1: {macro_f1:.4f}"
        )

        results.append({
            "Model": "kNN",
            "Training Samples": n_samples,
            "Accuracy": accuracy,
            "Error": error,
            "Macro-F1": macro_f1
        })

        # ----------------------------------------------------
        # SVM
        # ----------------------------------------------------

        if n_samples <= SVM_MAX_SAMPLES:

            print("Running SVM...")

            accuracy, error, macro_f1 = evaluate_svm(
                X_sample_pca,
                y_sample,
                X_test_pca,
                y_test
            )

            print(
                f"SVM       Accuracy: {accuracy:.4f} "
                f"Error: {error:.4f} "
                f"Macro-F1: {macro_f1:.4f}"
            )

            results.append({
                "Model": "SVM",
                "Training Samples": n_samples,
                "Accuracy": accuracy,
                "Error": error,
                "Macro-F1": macro_f1
            })

        else:

            print(
                "SVM       Skipped "
                "(sample size > 2000)"
            )

        # ----------------------------------------------------
        # Decision Tree
        # ----------------------------------------------------

        print("Running Decision Tree...")

        accuracy, error, macro_f1 = evaluate_tree(
            X_sample,
            y_sample,
            X_test,
            y_test
        )

        print(
            f"Decision Tree "
            f"Accuracy: {accuracy:.4f} "
            f"Error: {error:.4f} "
            f"Macro-F1: {macro_f1:.4f}"
        )

        results.append({
            "Model": "Decision Tree",
            "Training Samples": n_samples,
            "Accuracy": accuracy,
            "Error": error,
            "Macro-F1": macro_f1
        })

        # ----------------------------------------------------
        # Gradient Boosting
        # ----------------------------------------------------

        print("Running Gradient Boosting...")

        (
            accuracy,
            error,
            macro_f1,
            gb_model
        ) = evaluate_gradient_boosting(
            X_sample,
            y_sample,
            X_test,
            y_test
        )

        print(
            f"Gradient Boosting "
            f"Accuracy: {accuracy:.4f} "
            f"Error: {error:.4f} "
            f"Macro-F1: {macro_f1:.4f}"
        )

        results.append({
            "Model": "Gradient Boosting",
            "Training Samples": n_samples,
            "Accuracy": accuracy,
            "Error": error,
            "Macro-F1": macro_f1
        })

    return pd.DataFrame(results)


# ============================================================
# SWITCH RULE
# ============================================================

def run_switch_rule(
    X_train,
    y_train,
    X_test,
    y_test,
    switch_point
):
    """
    Combination 1.

    If training sample size is below switch_point:
        use Gradient Boosting

    Otherwise:
        use kNN

    The switch point is determined from the individual
    model comparison results.
    """

    results = []

    for n_samples in SAMPLE_SIZES:

        X_sample, y_sample = sample_training_data(
            X_train,
            y_train,
            n_samples
        )

        (
            X_sample_pca,
            X_test_pca,
            scaler,
            pca,
            actual_components
        ) = prepare_pca_data(
            X_sample,
            X_test,
            PCA_COMPONENTS
        )

        if n_samples < switch_point:

            selected_model = "Gradient Boosting"

            (
                accuracy,
                error,
                macro_f1,
                model
            ) = evaluate_gradient_boosting(
                X_sample,
                y_sample,
                X_test,
                y_test
            )

        else:

            selected_model = "kNN"

            accuracy, error, macro_f1 = evaluate_knn(
                X_sample_pca,
                y_sample,
                X_test_pca,
                y_test,
                k=KNN_K
            )

        print(
            f"Switch rule | "
            f"{n_samples} samples | "
            f"{selected_model} | "
            f"Accuracy: {accuracy:.4f} | "
            f"Error: {error:.4f}"
        )

        results.append({
            "Combination": "Switch Rule",
            "Training Samples": n_samples,
            "Selected Model": selected_model,
            "Accuracy": accuracy,
            "Error": error,
            "Macro-F1": macro_f1
        })

    return pd.DataFrame(results)


# ============================================================
# VOTING EXPERIMENT
# ============================================================

def run_voting_experiment(
    X_train,
    y_train,
    X_test,
    y_test
):

    results = []

    print("\n" + "=" * 70)
    print("PROBABILITY VOTING: kNN + GRADIENT BOOSTING")
    print("=" * 70)

    for n_samples in SAMPLE_SIZES:

        print(
            f"\nVoting with {n_samples} samples..."
        )

        X_sample, y_sample = sample_training_data(
            X_train,
            y_train,
            n_samples
        )

        # Voting operates on original standardized features.
        (
            X_sample_pca,
            X_test_pca,
            scaler,
            pca,
            actual_components
        ) = prepare_pca_data(
            X_sample,
            X_test,
            PCA_COMPONENTS
        )

        accuracy, error, macro_f1 = (
            evaluate_probability_voting(
                X_sample_pca,
                y_sample,
                X_test_pca,
                y_test
            )
        )

        print(
            f"Voting     Accuracy: {accuracy:.4f} "
            f"Error: {error:.4f} "
            f"Macro-F1: {macro_f1:.4f}"
        )

        results.append({
            "Combination": "kNN + Gradient Boosting Voting",
            "Training Samples": n_samples,
            "Selected Model": "kNN(k=3) + GB",
            "Accuracy": accuracy,
            "Error": error,
            "Macro-F1": macro_f1
        })

    return pd.DataFrame(results)


# ============================================================
# FIND SWITCH POINT
# ============================================================

def find_switch_point(results_df):
    """
    Find the first sample size where kNN error becomes
    lower than Gradient Boosting error.

    If no crossover occurs, use 1000 as the default
    based on the project paper's observation.
    """

    pivot = results_df.pivot_table(
        index="Training Samples",
        columns="Model",
        values="Error"
    )

    if (
        "kNN" not in pivot.columns
        or "Gradient Boosting" not in pivot.columns
    ):
        return 1000

    for sample_size in SAMPLE_SIZES:

        if sample_size not in pivot.index:
            continue

        knn_error = pivot.loc[
            sample_size,
            "kNN"
        ]

        gb_error = pivot.loc[
            sample_size,
            "Gradient Boosting"
        ]

        if (
            not pd.isna(knn_error)
            and not pd.isna(gb_error)
            and knn_error <= gb_error
        ):
            return sample_size

    return 1000


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    model_results,
    switch_results,
    voting_results
):

    main_path = os.path.join(
        TABLES_DIR,
        "model_comparison_results.csv"
    )

    combinations_path = os.path.join(
        TABLES_DIR,
        "model_comparison_combinations.csv"
    )

    model_results.to_csv(
        main_path,
        index=False
    )

    combinations = pd.concat(
        [
            switch_results,
            voting_results
        ],
        ignore_index=True
    )

    combinations.to_csv(
        combinations_path,
        index=False
    )

    return main_path, combinations_path


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STEP 10 - BIG MODEL COMPARISON")
    print("=" * 70)

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
    # ACTIVE WAPS
    # --------------------------------------------------------

    print(
        "\nSelecting active WAP features..."
    )

    active_waps = get_active_wap_columns(
        train_df
    )

    print(
        f"Active WAP features: {len(active_waps)}"
    )

    # --------------------------------------------------------
    # BUILDING 0
    # --------------------------------------------------------

    print(
        f"\nSelecting Building {BUILDING_ID}..."
    )

    X_train, y_train = split_by_building(
        train_df,
        active_waps,
        BUILDING_ID
    )

    X_test, y_test = split_by_building(
        validation_df,
        active_waps,
        BUILDING_ID
    )

    X_train = np.asarray(
        X_train,
        dtype=float
    )

    X_test = np.asarray(
        X_test,
        dtype=float
    )

    y_train = np.asarray(
        y_train
    )

    y_test = np.asarray(
        y_test
    )

    print(
        f"Building {BUILDING_ID} training samples: "
        f"{len(X_train)}"
    )

    print(
        f"Building {BUILDING_ID} test samples: "
        f"{len(X_test)}"
    )

    # --------------------------------------------------------
    # MAIN COMPARISON
    # --------------------------------------------------------

    model_results = run_model_comparison(
        X_train,
        y_train,
        X_test,
        y_test
    )

    # --------------------------------------------------------
    # SWITCH POINT
    # --------------------------------------------------------

    switch_point = find_switch_point(
        model_results
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "SWITCH POINT"
    )

    print(
        "=" * 70
    )

    print(
        f"Selected switch point: {switch_point} samples"
    )

    # --------------------------------------------------------
    # SWITCH RULE
    # --------------------------------------------------------

    switch_results = run_switch_rule(
        X_train,
        y_train,
        X_test,
        y_test,
        switch_point
    )

    # --------------------------------------------------------
    # PROBABILITY VOTING
    # --------------------------------------------------------

    voting_results = run_voting_experiment(
        X_train,
        y_train,
        X_test,
        y_test
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    main_path, combinations_path = save_results(
        model_results,
        switch_results,
        voting_results
    )

    # --------------------------------------------------------
    # PRINT FINAL TABLE
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL MODEL COMPARISON"
    )

    print(
        "=" * 70
    )

    print(
        model_results.to_string(
            index=False
        )
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "COMBINATION RESULTS"
    )

    print(
        "=" * 70
    )

    combinations = pd.concat(
        [
            switch_results,
            voting_results
        ],
        ignore_index=True
    )

    print(
        combinations.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # BEST MODELS
    # --------------------------------------------------------

    best_model_row = model_results.loc[
        model_results["Accuracy"].idxmax()
    ]

    best_combination_row = combinations.loc[
        combinations["Accuracy"].idxmax()
    ]

    print(
        "\n" + "=" * 70
    )

    print(
        "BEST RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"Best individual model: "
        f"{best_model_row['Model']}"
    )

    print(
        f"Training samples: "
        f"{int(best_model_row['Training Samples'])}"
    )

    print(
        f"Accuracy: "
        f"{best_model_row['Accuracy']:.4f}"
    )

    print(
        f"Macro-F1: "
        f"{best_model_row['Macro-F1']:.4f}"
    )

    print()

    print(
        f"Best combination: "
        f"{best_combination_row['Combination']}"
    )

    print(
        f"Training samples: "
        f"{int(best_combination_row['Training Samples'])}"
    )

    print(
        f"Accuracy: "
        f"{best_combination_row['Accuracy']:.4f}"
    )

    print(
        f"Macro-F1: "
        f"{best_combination_row['Macro-F1']:.4f}"
    )

    # --------------------------------------------------------
    # FILES
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "FILES SAVED"
    )

    print(
        "=" * 70
    )

    print(
        f"Model comparison:"
        f"\n{main_path}"
    )

    print(
        f"\nCombinations:"
        f"\n{combinations_path}"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "MODEL COMPARISON COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()