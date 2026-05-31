import numpy as np


def safe_norm(v: np.ndarray, min_val: float = 1e-12) -> float:
    """Scalar norm of a 1D vector, clamped to avoid division by zero."""
    return max(float(np.linalg.norm(v)), min_val)


def pairwise_displacements(positions: np.ndarray) -> np.ndarray:
    """
    Returns dr[i, j] = positions[j] - positions[i], shape (N, N, D).
    Fully vectorized; no Python loops.
    """
    return positions[np.newaxis, :, :] - positions[:, np.newaxis, :]


def pairwise_distances(positions: np.ndarray, epsilon: float = 0.0) -> np.ndarray:
    """
    Returns scalar distances r[i,j], shape (N, N).
    Softening epsilon prevents division by zero on the diagonal.
    """
    dr = pairwise_displacements(positions)
    r2 = np.einsum('ijk,ijk->ij', dr, dr) + epsilon ** 2
    return np.sqrt(r2)
