import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


def get_pruning_path(X, y, random_state=42):
    """
    Compute the cost-complexity pruning path for a decision tree.

    Args:
        X: Feature matrix.
        y: Target labels.
        random_state: Seed for reproducible tree construction.

    Returns:
        Tuple containing candidate pruning strengths and
        corresponding total leaf impurities.
    """
    tree = DecisionTreeClassifier(
        random_state=random_state
    )

    tree.fit(X, y)

    path = tree.cost_complexity_pruning_path(X, y)

    return path.ccp_alphas, path.impurities


def evaluate_pruning_alphas(
    X,
    y,
    ccp_alphas,
    cv_splits=10,
    random_state=42
):
    """
    Evaluate pruning strengths using stratified K-fold cross-validation.

    Args:
        X: Feature matrix.
        y: Target labels.
        ccp_alphas: Candidate cost-complexity pruning strengths.
        cv_splits: Number of stratified cross-validation folds.
        random_state: Seed for reproducible splitting.

    Returns:
        Tuple containing mean CV error and standard error for each alpha.
    """
    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=random_state
    )

    means = []
    ses = []

    for alpha in ccp_alphas:
        tree = DecisionTreeClassifier(
            ccp_alpha=alpha,
            random_state=random_state
        )

        scores = cross_val_score(
            tree,
            X,
            y,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1
        )

        errors = 1.0 - scores

        mean_error = np.mean(errors)

        # Standard error = standard deviation / sqrt(number of folds).
        standard_error = np.std(
            errors,
            ddof=1
        ) / np.sqrt(len(errors))

        means.append(mean_error)
        ses.append(standard_error)

    return np.array(means), np.array(ses)


def choose_alpha(
    ccp_alphas,
    means,
    ses,
    rule="min"
):
    """
    Select a pruning alpha using the minimum-error or 1-SE rule.

    Args:
        ccp_alphas: Candidate pruning strengths.
        means: Mean CV error for each alpha.
        ses: Standard error for each alpha.
        rule: Selection rule, either "min" or "1se".

    Returns:
        Selected pruning alpha.

    Raises:
        ValueError: If rule is not "min" or "1se".
    """
    ccp_alphas = np.asarray(ccp_alphas)
    means = np.asarray(means)
    ses = np.asarray(ses)

    best_index = np.argmin(means)

    if rule == "min":
        return ccp_alphas[best_index]

    elif rule == "1se":
        best_error = means[best_index]
        best_se = ses[best_index]

        threshold = best_error + best_se

        valid = np.where(
            means <= threshold
        )[0]

        # Largest alpha corresponds to the simplest tree.
        return ccp_alphas[valid[-1]]

    else:
        raise ValueError(
            "rule must be either 'min' or '1se'"
        )


def train_tree(
    X,
    y,
    ccp_alpha=0.0,
    random_state=42
):
    """
    Train a decision tree using the supplied pruning strength.

    Args:
        X: Feature matrix.
        y: Target labels.
        ccp_alpha: Cost-complexity pruning strength.
        random_state: Seed for reproducible tree construction.

    Returns:
        Trained DecisionTreeClassifier.
    """
    tree = DecisionTreeClassifier(
        ccp_alpha=ccp_alpha,
        random_state=random_state
    )

    tree.fit(X, y)

    return tree


def tree_statistics(tree):
    """
    Get basic structural statistics from a trained decision tree.

    Args:
        tree: Trained DecisionTreeClassifier.

    Returns:
        Dictionary containing tree depth, number of leaves,
        and total number of nodes.
    """
    return {
        "depth": tree.get_depth(),
        "leaves": tree.get_n_leaves(),
        "nodes": tree.tree_.node_count
    }


def get_feature_importances(tree):
    """
    Get feature importance values from a trained decision tree.

    Args:
        tree: Trained DecisionTreeClassifier.

    Returns:
        Array containing the importance of each feature.
    """
    return tree.feature_importances_


def get_top_features(
    tree,
    feature_names,
    n=5
):
    """
    Get the top n non-zero feature importances.

    Args:
        tree: Trained DecisionTreeClassifier.
        feature_names: Names corresponding to the tree features.
        n: Maximum number of features to return.

    Returns:
        List of tuples containing feature names and importance values.
    """
    importances = tree.feature_importances_

    indices = np.argsort(importances)[::-1]

    results = []

    for index in indices:
        if importances[index] <= 0:
            continue

        results.append(
            (
                feature_names[index],
                importances[index]
            )
        )

        if len(results) == n:
            break

    return results