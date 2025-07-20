import numpy as np
from scipy import linalg
from config import Config

def ensure_positive_definite(matrix, min_eigenvalue=None):
    """Ensure matrix is positive definite by adding a small diagonal shift if needed."""
    if min_eigenvalue is None:
        min_eigenvalue = Config.MATRIX_OPERATIONS['numerical_stability']['min_eigenvalue']
    
    # Compute eigenvalues
    eigenvals = linalg.eigvalsh(matrix)
    min_val = np.min(eigenvals)
    
    # If matrix is not positive definite, add a small shift
    if min_val < min_eigenvalue:
        shift = min_eigenvalue - min_val + Config.MATRIX_OPERATIONS['numerical_stability']['regularization_factor']
        matrix = matrix + shift * np.eye(matrix.shape[0])
    
    return matrix

def stable_cholesky(matrix, lower=True):
    """Perform numerically stable Cholesky decomposition."""
    # Get configuration
    cholesky_config = Config.MATRIX_OPERATIONS['cholesky']
    
    # Ensure matrix is positive definite
    matrix = ensure_positive_definite(matrix)
    
    try:
        # Attempt Cholesky decomposition with configured settings
        L = linalg.cholesky(matrix, lower=lower)
        return L
    except linalg.LinAlgError:
        # If failed, try with increased regularization
        matrix = matrix + cholesky_config['shift'] * np.eye(matrix.shape[0])
        return linalg.cholesky(matrix, lower=lower)

def condition_number(matrix):
    """Compute the condition number of a matrix."""
    return np.linalg.cond(matrix)

def is_well_conditioned(matrix):
    """Check if matrix is well-conditioned."""
    cond = condition_number(matrix)
    return cond < Config.MATRIX_OPERATIONS['numerical_stability']['condition_number_threshold']

def regularize_matrix(matrix, factor=None):
    """Add regularization to improve matrix conditioning."""
    if factor is None:
        factor = Config.MATRIX_OPERATIONS['numerical_stability']['regularization_factor']
    
    return matrix + factor * np.eye(matrix.shape[0]) 