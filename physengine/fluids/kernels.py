"""
Smoothed Particle Hydrodynamics (SPH) kernel functions.

SPH approximates a continuous field f(r) as a weighted sum over
nearby particles using a kernel W(r, h):

    f(r) ≈ sum_j  m_j / rho_j  * f(r_j) * W(|r - r_j|, h)

The kernel must:
  1. Be normalized: integral W(r, h) dr = 1
  2. Have compact support: W(r, h) = 0 for r > kh
  3. Approach a delta function as h -> 0

This module provides popular kernel choices and their gradients.
"""
import numpy as np


def cubic_spline_kernel(r: np.ndarray, h: float, dim: int = 2) -> np.ndarray:
    """
    Cubic spline kernel (Monaghan 1992).

    Compact support at 2h. Widely used for its smoothness and
    good stability properties.

    Parameters
    ----------
    r   : array of scalar distances |r_i - r_j|
    h   : smoothing length
    dim : spatial dimension (1, 2, or 3)

    Returns
    -------
    W   : kernel values, same shape as r
    """
    q = np.asarray(r) / h

    # Normalization constants
    sigma = {1: 2.0 / 3.0, 2: 10.0 / (7.0 * np.pi), 3: 1.0 / np.pi}
    alpha = sigma[dim] / h ** dim

    W = np.zeros_like(q, dtype=np.float64)

    mask1 = (q >= 0) & (q < 1)
    mask2 = (q >= 1) & (q < 2)

    W[mask1] = alpha * (1.0 - 1.5 * q[mask1] ** 2 + 0.75 * q[mask1] ** 3)
    W[mask2] = alpha * 0.25 * (2.0 - q[mask2]) ** 3

    return W


def cubic_spline_gradient_factor(r: np.ndarray, h: float, dim: int = 2) -> np.ndarray:
    """
    Returns dW/dr divided by r (i.e., the factor to multiply by (r_i - r_j) 
    to get the gradient vector).

    grad_i W_ij = (dW/dr) * (r_i - r_j) / |r_i - r_j|
                = gradient_factor(|r_ij|, h) * (r_i - r_j)

    Parameters
    ----------
    r   : scalar distances
    h   : smoothing length
    dim : spatial dimension

    Returns
    -------
    factor : dW/dr / r, same shape as r
    """
    q = np.asarray(r) / h

    sigma = {1: 2.0 / 3.0, 2: 10.0 / (7.0 * np.pi), 3: 1.0 / np.pi}
    alpha = sigma[dim] / h ** dim

    dWdq = np.zeros_like(q, dtype=np.float64)

    mask1 = (q >= 0) & (q < 1)
    mask2 = (q >= 1) & (q < 2)

    dWdq[mask1] = alpha * (-3.0 * q[mask1] + 2.25 * q[mask1] ** 2)
    dWdq[mask2] = alpha * (-0.75 * (2.0 - q[mask2]) ** 2)

    # dW/dr = dW/dq * 1/h, and we want dW/dr / r = dW/dq / (h * r)
    # But r = q * h, so dW/dr / r = dW/dq / (h * q * h) = dW/dq / (h^2 * q)
    # Guard against q = 0
    safe_q = np.where(q > 1e-12, q, 1e-12)
    factor = dWdq / (h ** 2 * safe_q)

    return factor


def wendland_c2_kernel(r: np.ndarray, h: float, dim: int = 2) -> np.ndarray:
    """
    Wendland C2 kernel (Wendland 1995).

    Compact support at 2h. Positive definite — avoids the tensile
    instability that can affect cubic spline kernels.

    Parameters
    ----------
    r   : scalar distances
    h   : smoothing length
    dim : spatial dimension (2 or 3)

    Returns
    -------
    W   : kernel values
    """
    q = np.asarray(r) / h

    sigma = {2: 7.0 / (4.0 * np.pi), 3: 21.0 / (16.0 * np.pi)}
    alpha = sigma[dim] / h ** dim

    W = np.zeros_like(q, dtype=np.float64)
    mask = (q >= 0) & (q < 2)
    qm = q[mask]
    W[mask] = alpha * (1.0 - 0.5 * qm) ** 4 * (2.0 * qm + 1.0)

    return W


def wendland_c2_gradient_factor(r: np.ndarray, h: float, dim: int = 2) -> np.ndarray:
    """
    Gradient factor for Wendland C2 kernel (dW/dr / r).
    """
    q = np.asarray(r) / h

    sigma = {2: 7.0 / (4.0 * np.pi), 3: 21.0 / (16.0 * np.pi)}
    alpha = sigma[dim] / h ** dim

    dWdq = np.zeros_like(q, dtype=np.float64)
    mask = (q >= 0) & (q < 2)
    qm = q[mask]
    # d/dq [(1 - q/2)^4 (2q + 1)] = -5q(1 - q/2)^3
    dWdq[mask] = alpha * (-5.0 * qm * (1.0 - 0.5 * qm) ** 3)

    safe_q = np.where(q > 1e-12, q, 1e-12)
    factor = dWdq / (h ** 2 * safe_q)

    return factor
