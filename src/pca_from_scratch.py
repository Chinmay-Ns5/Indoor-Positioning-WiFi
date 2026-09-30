import numpy as np


def standardize(X):
    """
    Standardize each feature to zero mean and unit variance.

    Args:
        X: Input feature matrix.

    Returns:
        Tuple containing the standardized data, feature means,
        and feature standard deviations.
    """
    X = np.asarray(X, dtype=float)

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    # Prevent division by zero for constant features.
    std[std == 0] = 1.0

    X_standardized = (X - mean) / std

    return X_standardized, mean, std


def compute_covariance_matrix(X):
    """
    Compute the covariance matrix of standardized data.

    Args:
        X: Standardized feature matrix.

    Returns:
        Covariance matrix of the input features.
    """
    n_samples = X.shape[0]

    covariance_matrix = (
        X.T @ X
    ) / (n_samples - 1)

    return covariance_matrix


def compute_eigenpairs(covariance_matrix):
    """
    Compute and sort eigenvalues and eigenvectors.

    Args:
        covariance_matrix: Covariance matrix of the data.

    Returns:
        Tuple containing eigenvalues and corresponding eigenvectors,
        sorted from largest to smallest eigenvalue.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance_matrix
    )

    # Sort eigenvalues from largest to smallest.
    indices = np.argsort(eigenvalues)[::-1]

    eigenvalues = eigenvalues[indices]
    eigenvectors = eigenvectors[:, indices]

    return eigenvalues, eigenvectors


def explained_variance_ratio(eigenvalues):
    """
    Calculate the explained variance ratio for each component.

    Args:
        eigenvalues: Eigenvalues of the covariance matrix.

    Returns:
        Array containing the explained variance ratio of each component.
    """
    return eigenvalues / np.sum(eigenvalues)


def cumulative_explained_variance(eigenvalues):
    """
    Calculate cumulative explained variance across components.

    Args:
        eigenvalues: Eigenvalues of the covariance matrix.

    Returns:
        Array containing cumulative explained variance ratios.
    """
    ratios = explained_variance_ratio(eigenvalues)

    return np.cumsum(ratios)


def project_data(X, eigenvectors, n_components):
    """
    Project data onto the first n principal components.

    Args:
        X: Standardized input feature matrix.
        eigenvectors: Eigenvectors sorted by descending eigenvalue.
        n_components: Number of principal components to retain.

    Returns:
        Data projected onto the selected principal components.
    """
    components = eigenvectors[:, :n_components]

    return X @ components


def reconstruct_data(X_projected, eigenvectors, n_components):
    """
    Reconstruct standardized data from selected components.

    Args:
        X_projected: Data projected onto the selected components.
        eigenvectors: Eigenvectors sorted by descending eigenvalue.
        n_components: Number of components used for reconstruction.

    Returns:
        Reconstructed standardized feature matrix.
    """
    components = eigenvectors[:, :n_components]

    return X_projected @ components.T