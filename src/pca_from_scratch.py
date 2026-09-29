import numpy as np


def standardize(X):
    """
    Standardize each feature to zero mean and unit variance.
    """
    X = np.asarray(X, dtype=float)

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    # Prevent division by zero
    std[std == 0] = 1.0

    X_standardized = (X - mean) / std

    return X_standardized, mean, std


def compute_covariance_matrix(X):
    """
    Compute the covariance matrix of standardized data.
    """
    n_samples = X.shape[0]

    covariance_matrix = (
        X.T @ X
    ) / (n_samples - 1)

    return covariance_matrix


def compute_eigenpairs(covariance_matrix):
    """
    Compute eigenvalues and eigenvectors of
    the covariance matrix.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance_matrix
    )

    # Sort eigenvalues from largest to smallest
    indices = np.argsort(eigenvalues)[::-1]

    eigenvalues = eigenvalues[indices]
    eigenvectors = eigenvectors[:, indices]

    return eigenvalues, eigenvectors


def explained_variance_ratio(eigenvalues):
    """
    Calculate the explained variance ratio
    for each principal component.
    """
    return eigenvalues / np.sum(eigenvalues)


def cumulative_explained_variance(eigenvalues):
    """
    Calculate cumulative explained variance.
    """
    ratios = explained_variance_ratio(eigenvalues)

    return np.cumsum(ratios)


def project_data(X, eigenvectors, n_components):
    """
    Project data onto the first n principal components.
    """
    components = eigenvectors[:, :n_components]

    return X @ components


def reconstruct_data(X_projected, eigenvectors, n_components):
    """
    Reconstruct standardized data from the selected
    principal components.
    """
    components = eigenvectors[:, :n_components]

    return X_projected @ components.T