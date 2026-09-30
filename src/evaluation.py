import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import RepeatedStratifiedKFold


def calculate_metrics(y_true, y_pred):
    """
    Calculate accuracy and F1 classification metrics.

    Args:
        y_true: True class labels.
        y_pred: Predicted class labels.

    Returns:
        Dictionary containing accuracy, macro-F1, and weighted-F1 scores.
    """
    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro"
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted"
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def repeated_cv_error(
    X,
    y,
    model_factory,
    n_splits=10,
    n_repeats=5,
    random_state=42,
):
    """
    Evaluate a model using repeated stratified cross-validation.

    Args:
        X: Feature matrix.
        y: Target labels.
        model_factory: Function that creates a fresh model for each fold.
        n_splits: Number of folds in each cross-validation repetition.
        n_repeats: Number of cross-validation repetitions.
        random_state: Seed used for reproducible splitting.

    Returns:
        Dictionary containing fold-level accuracy and macro-F1 scores,
        their means and standard deviations, the aggregated confusion
        matrix, and the corresponding class labels.
    """
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )

    accuracies = []
    macro_f1_scores = []

    all_true = []
    all_pred = []

    for fold_number, (train_idx, test_idx) in enumerate(
        cv.split(X, y),
        start=1
    ):
        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        # Create a fresh model for every fold.
        model = model_factory()

        model.fit(
            X_train,
            y_train
        )

        y_pred = model.predict(
            X_test
        )

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

        all_true.extend(
            y_test
        )

        all_pred.extend(
            y_pred
        )

        print(
            f"Fold {fold_number:02d}: "
            f"Accuracy = {accuracy:.4f}, "
            f"Macro-F1 = {macro_f1:.4f}"
        )

    accuracies = np.array(
        accuracies
    )

    macro_f1_scores = np.array(
        macro_f1_scores
    )

    labels = np.unique(y)

    cm = confusion_matrix(
        all_true,
        all_pred,
        labels=labels
    )

    results = {
        "accuracies": accuracies,
        "macro_f1_scores": macro_f1_scores,
        "accuracy_mean": accuracies.mean(),
        "accuracy_std": accuracies.std(),
        "macro_f1_mean": macro_f1_scores.mean(),
        "macro_f1_std": macro_f1_scores.std(),
        "confusion_matrix": cm,
        "labels": labels,
    }

    return results