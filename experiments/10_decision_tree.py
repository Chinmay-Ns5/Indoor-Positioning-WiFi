"""
Decision Tree Experiments
Indoor Positioning Using WiFi Fingerprints

Experiments:
1. Full decision tree on raw RSSI features
2. Cost-complexity pruning
3. 10-fold cross-validation for pruning
4. Minimum-CV pruning
5. One-standard-error (1-SE) pruning
6. Tree comparison
7. Top-5 important WAPs
8. Save results and plots

Run:
    python -m experiments.10_decision_tree
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.data_utils import (
    load_data,
    get_active_wap_columns,
    split_by_building,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data" / "trainingData.csv"
VALIDATION_PATH = ROOT / "data" / "validationData.csv"

RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)
TABLES_DIR.mkdir(exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

BUILDING_ID = 0

RANDOM_STATE = 42

N_FOLDS = 10

# We use the complete building-0 training data.
# This is consistent with the project experiment.
MAX_ALPHA_CANDIDATES = 80


# ============================================================
# HELPER: SAFE ERROR
# ============================================================

def classification_error(y_true, y_pred):
    """
    Classification error = 1 - accuracy.
    """
    return 1.0 - accuracy_score(y_true, y_pred)


# ============================================================
# HELPER: TRAIN TREE
# ============================================================

def train_tree(X, y, ccp_alpha=0.0):
    """
    Train a decision tree with a given pruning strength.
    """

    model = DecisionTreeClassifier(
        criterion="gini",
        random_state=RANDOM_STATE,
        ccp_alpha=ccp_alpha
    )

    model.fit(X, y)

    return model


# ============================================================
# GET PRUNING PATH
# ============================================================

def get_pruning_path(X, y):
    """
    Get cost-complexity pruning alphas from a fully grown tree.
    """

    print("\nComputing cost-complexity pruning path...")

    full_tree = DecisionTreeClassifier(
        criterion="gini",
        random_state=RANDOM_STATE,
        ccp_alpha=0.0
    )

    path = full_tree.cost_complexity_pruning_path(X, y)

    alphas = path.ccp_alphas

    # The final alpha normally produces a tree with one node.
    # It is not useful for the pruning comparison, so remove it.
    if len(alphas) > 1:
        alphas = alphas[:-1]

    print(f"Total pruning alphas found: {len(alphas)}")

    return alphas


# ============================================================
# REDUCE ALPHA COUNT IF NECESSARY
# ============================================================

def select_alpha_candidates(alphas):
    """
    Keep the pruning experiment computationally reasonable.

    If the pruning path is small, use every alpha.

    If it is very large, select evenly spaced candidates while
    preserving the minimum and maximum useful pruning levels.
    """

    alphas = np.asarray(alphas, dtype=float)

    # Remove duplicates
    alphas = np.unique(alphas)

    if len(alphas) <= MAX_ALPHA_CANDIDATES:
        return alphas

    print(
        f"Pruning path contains {len(alphas)} values."
    )

    print(
        f"Using {MAX_ALPHA_CANDIDATES} representative "
        f"alpha values to keep 10-fold CV practical."
    )

    indices = np.linspace(
        0,
        len(alphas) - 1,
        MAX_ALPHA_CANDIDATES,
        dtype=int
    )

    selected = alphas[indices]

    return np.unique(selected)


# ============================================================
# CROSS-VALIDATION PRUNING
# ============================================================

def pruning_curve(X, y, alphas):
    """
    Evaluate each pruning strength using 10-fold CV.

    Returns:
        means
        standard errors
    """

    print("\n" + "=" * 60)
    print("10-FOLD CROSS-VALIDATION FOR PRUNING")
    print("=" * 60)

    skf = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    mean_errors = []
    standard_errors = []

    for index, alpha in enumerate(alphas, start=1):

        fold_errors = []

        print(
            f"\nAlpha {index}/{len(alphas)} "
            f"= {alpha:.8f}"
        )

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(X, y),
            start=1
        ):

            X_train = X[train_idx]
            X_val = X[val_idx]

            y_train = y[train_idx]
            y_val = y[val_idx]

            model = train_tree(
                X_train,
                y_train,
                ccp_alpha=alpha
            )

            predictions = model.predict(X_val)

            error = classification_error(
                y_val,
                predictions
            )

            fold_errors.append(error)

        fold_errors = np.asarray(fold_errors)

        mean_error = np.mean(fold_errors)

        # Standard error of the mean
        standard_error = (
            np.std(fold_errors, ddof=1)
            / np.sqrt(N_FOLDS)
        )

        mean_errors.append(mean_error)
        standard_errors.append(standard_error)

        print(
            f"CV Error = {mean_error:.6f} "
            f"+/- {standard_error:.6f}"
        )

    return (
        np.asarray(mean_errors),
        np.asarray(standard_errors)
    )


# ============================================================
# CHOOSE ALPHA
# ============================================================

def choose_alpha(alphas, means, ses, rule="min"):
    """
    Choose pruning alpha.

    rule="min":
        Select alpha with minimum CV error.

    rule="1se":
        Select the largest alpha whose error is within
        one standard error of the minimum-error model.
    """

    alphas = np.asarray(alphas)
    means = np.asarray(means)
    ses = np.asarray(ses)

    best_index = np.argmin(means)

    best_error = means[best_index]

    best_se = ses[best_index]

    if rule == "min":

        return alphas[best_index]

    elif rule == "1se":

        threshold = best_error + best_se

        eligible = alphas[means <= threshold]

        if len(eligible) == 0:
            return alphas[best_index]

        # Largest alpha = simplest tree
        return np.max(eligible)

    else:

        raise ValueError(
            "rule must be either 'min' or '1se'"
        )


# ============================================================
# TREE INFORMATION
# ============================================================

def tree_statistics(model, X, y):
    """
    Return useful statistics for a trained tree.
    """

    predictions = model.predict(X)

    accuracy = accuracy_score(
        y,
        predictions
    )

    macro_f1 = f1_score(
        y,
        predictions,
        average="macro"
    )

    return {
        "depth": model.get_depth(),
        "leaves": model.get_n_leaves(),
        "accuracy": accuracy,
        "error": 1.0 - accuracy,
        "macro_f1": macro_f1,
    }


# ============================================================
# TOP FEATURES
# ============================================================

def get_top_features(model, feature_names, n=5):
    """
    Return the top-n most important WAP features.
    """

    importances = model.feature_importances_

    order = np.argsort(importances)[::-1]

    top_features = []

    for index in order:

        if importances[index] <= 0:
            break

        top_features.append(
            (
                feature_names[index],
                importances[index]
            )
        )

        if len(top_features) == n:
            break

    return top_features


# ============================================================
# SAVE TOP FEATURES
# ============================================================

def save_top_features(
    full_tree,
    min_tree,
    one_se_tree,
    feature_names
):

    rows = []

    tree_info = [
        ("Full Tree", full_tree),
        ("Minimum CV Tree", min_tree),
        ("1-SE Tree", one_se_tree),
    ]

    for tree_name, model in tree_info:

        top_features = get_top_features(
            model,
            feature_names,
            n=5
        )

        for rank, (feature, importance) in enumerate(
            top_features,
            start=1
        ):

            rows.append(
                {
                    "Tree": tree_name,
                    "Rank": rank,
                    "Feature": feature,
                    "Importance": importance
                }
            )

    df = pd.DataFrame(rows)

    output_path = (
        TABLES_DIR /
        "decision_tree_top_features.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    return df


# ============================================================
# PLOT PRUNING CURVE
# ============================================================

def plot_pruning_curve(
    alphas,
    mean_errors,
    standard_errors,
    alpha_min,
    alpha_1se
):

    plt.figure(figsize=(9, 6))

    plt.errorbar(
        alphas,
        mean_errors,
        yerr=standard_errors,
        fmt="o-",
        markersize=3,
        capsize=3
    )

    plt.axvline(
        alpha_min,
        linestyle="--",
        label="Minimum CV"
    )

    plt.axvline(
        alpha_1se,
        linestyle="--",
        label="1-SE"
    )

    plt.xlabel("ccp_alpha")

    plt.ylabel("10-Fold CV Error")

    plt.title(
        "Decision Tree Cost-Complexity Pruning"
    )

    plt.legend()

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = (
        FIGURES_DIR /
        "decision_tree_pruning_curve.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    return output_path


# ============================================================
# PLOT TREE COMPARISON
# ============================================================

def plot_tree_comparison(results_df):

    plt.figure(figsize=(9, 6))

    names = results_df["Tree"].tolist()

    errors = results_df["CV Error"].tolist()

    plt.bar(
        names,
        errors
    )

    plt.ylabel("Error")

    plt.title(
        "Decision Tree Error Comparison"
    )

    plt.xticks(
        rotation=15
    )

    plt.tight_layout()

    output_path = (
        FIGURES_DIR /
        "decision_tree_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    return output_path


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("DECISION TREE EXPERIMENTS")
    print("=" * 60)

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

    print("\nSelecting active WAPs...")

    active_waps = get_active_wap_columns(
        train_df
    )

    print(
        f"Active WAP features: {len(active_waps)}"
    )

    if len(active_waps) == 0:

        raise RuntimeError(
            "No active WAP columns were found."
        )

    # --------------------------------------------------------
    # BUILDING 0
    # --------------------------------------------------------

    print(
        f"\nSelecting Building {BUILDING_ID}..."
    )

    # IMPORTANT:
    #
    # split_by_building() expects:
    #
    # split_by_building(df, wap_cols, building_id)
    #
    # Therefore we call it separately for the training
    # dataframe and validation dataframe.

    X_train_df, y_train_series = split_by_building(
        train_df,
        active_waps,
        BUILDING_ID
    )

    X_test_df, y_test_series = split_by_building(
        validation_df,
        active_waps,
        BUILDING_ID
    )

    # --------------------------------------------------------
    # CONVERT TO NUMPY
    # --------------------------------------------------------

    X_train = X_train_df.to_numpy(
        dtype=float
    )

    X_test = X_test_df.to_numpy(
        dtype=float
    )

    y_train = y_train_series.to_numpy()

    y_test = y_test_series.to_numpy()

    feature_names = list(
        X_train_df.columns
    )

    print(
        f"Training samples: {X_train.shape[0]}"
    )

    print(
        f"Test samples: {X_test.shape[0]}"
    )

    print(
        f"Features: {X_train.shape[1]}"
    )

    # --------------------------------------------------------
    # FULL TREE
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("TRAINING FULL DECISION TREE")
    print("=" * 60)

    full_tree = train_tree(
        X_train,
        y_train,
        ccp_alpha=0.0
    )

    full_stats = tree_statistics(
        full_tree,
        X_test,
        y_test
    )

    print(
        f"\nFull tree depth: "
        f"{full_tree.get_depth()}"
    )

    print(
        f"Full tree leaves: "
        f"{full_tree.get_n_leaves()}"
    )

    print(
        f"Test accuracy: "
        f"{full_stats['accuracy']:.4f}"
    )

    print(
        f"Test error: "
        f"{full_stats['error']:.4f}"
    )

    print(
        f"Macro-F1: "
        f"{full_stats['macro_f1']:.4f}"
    )

    # --------------------------------------------------------
    # PRUNING PATH
    # --------------------------------------------------------

    alphas = get_pruning_path(
        X_train,
        y_train
    )

    alphas = select_alpha_candidates(
        alphas
    )

    print(
        f"\nAlpha values used: "
        f"{len(alphas)}"
    )

    # --------------------------------------------------------
    # CROSS VALIDATION
    # --------------------------------------------------------

    mean_errors, standard_errors = pruning_curve(
        X_train,
        y_train,
        alphas
    )

    # --------------------------------------------------------
    # SELECT ALPHAS
    # --------------------------------------------------------

    alpha_min = choose_alpha(
        alphas,
        mean_errors,
        standard_errors,
        rule="min"
    )

    alpha_1se = choose_alpha(
        alphas,
        mean_errors,
        standard_errors,
        rule="1se"
    )

    min_index = np.argmin(
        np.abs(alphas - alpha_min)
    )

    print("\n" + "=" * 60)
    print("SELECTED PRUNING PARAMETERS")
    print("=" * 60)

    print(
        f"Minimum-CV alpha: "
        f"{alpha_min:.10f}"
    )

    print(
        f"Minimum CV error: "
        f"{mean_errors[min_index]:.6f}"
    )

    print(
        f"1-SE alpha: "
        f"{alpha_1se:.10f}"
    )

    # --------------------------------------------------------
    # MINIMUM-CV TREE
    # --------------------------------------------------------

    print("\nTraining minimum-CV pruned tree...")

    min_tree = train_tree(
        X_train,
        y_train,
        ccp_alpha=alpha_min
    )

    min_stats = tree_statistics(
        min_tree,
        X_test,
        y_test
    )

    # --------------------------------------------------------
    # 1-SE TREE
    # --------------------------------------------------------

    print("Training 1-SE pruned tree...")

    one_se_tree = train_tree(
        X_train,
        y_train,
        ccp_alpha=alpha_1se
    )

    one_se_stats = tree_statistics(
        one_se_tree,
        X_test,
        y_test
    )

    # --------------------------------------------------------
    # FINAL COMPARISON
    # --------------------------------------------------------

    results = pd.DataFrame(
        [
            {
                "Tree": "Full",
                "ccp_alpha": 0.0,
                "Depth": full_tree.get_depth(),
                "Leaves": full_tree.get_n_leaves(),
                "CV Error": np.nan,
                "Test Accuracy": full_stats["accuracy"],
                "Test Error": full_stats["error"],
                "Macro-F1": full_stats["macro_f1"],
            },
            {
                "Tree": "Minimum CV",
                "ccp_alpha": alpha_min,
                "Depth": min_tree.get_depth(),
                "Leaves": min_tree.get_n_leaves(),
                "CV Error": mean_errors[min_index],
                "Test Accuracy": min_stats["accuracy"],
                "Test Error": min_stats["error"],
                "Macro-F1": min_stats["macro_f1"],
            },
            {
                "Tree": "1-SE",
                "ccp_alpha": alpha_1se,
                "Depth": one_se_tree.get_depth(),
                "Leaves": one_se_tree.get_n_leaves(),
                "CV Error": mean_errors[
                    np.argmin(
                        np.abs(
                            alphas - alpha_1se
                        )
                    )
                ],
                "Test Accuracy": one_se_stats["accuracy"],
                "Test Error": one_se_stats["error"],
                "Macro-F1": one_se_stats["macro_f1"],
            },
        ]
    )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL DECISION TREE RESULTS")
    print("=" * 60)

    print(
        results.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    # --------------------------------------------------------
    # TOP FEATURES
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("TOP-5 IMPORTANT WAP FEATURES")
    print("=" * 60)

    top_features_df = save_top_features(
        full_tree,
        min_tree,
        one_se_tree,
        feature_names
    )

    print(
        top_features_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    # --------------------------------------------------------
    # SAVE PRUNING RESULTS
    # --------------------------------------------------------

    pruning_results = pd.DataFrame(
        {
            "ccp_alpha": alphas,
            "cv_error": mean_errors,
            "standard_error": standard_errors
        }
    )

    pruning_path = (
        TABLES_DIR /
        "decision_tree_pruning_results.csv"
    )

    pruning_results.to_csv(
        pruning_path,
        index=False
    )

    # --------------------------------------------------------
    # SAVE TREE RESULTS
    # --------------------------------------------------------

    tree_results_path = (
        TABLES_DIR /
        "decision_tree_results.csv"
    )

    results.to_csv(
        tree_results_path,
        index=False
    )

    # --------------------------------------------------------
    # PLOTS
    # --------------------------------------------------------

    pruning_plot = plot_pruning_curve(
        alphas,
        mean_errors,
        standard_errors,
        alpha_min,
        alpha_1se
    )

    comparison_plot = plot_tree_comparison(
        results
    )

    # --------------------------------------------------------
    # FINAL MESSAGE
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FILES SAVED")
    print("=" * 60)

    print(
        f"Tree results:\n"
        f"{tree_results_path}"
    )

    print(
        f"\nPruning results:\n"
        f"{pruning_path}"
    )

    print(
        f"\nTop features:\n"
        f"{TABLES_DIR / 'decision_tree_top_features.csv'}"
    )

    print(
        f"\nPruning plot:\n"
        f"{pruning_plot}"
    )

    print(
        f"\nComparison plot:\n"
        f"{comparison_plot}"
    )

    print("\n" + "=" * 60)
    print("DECISION TREE EXPERIMENT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()