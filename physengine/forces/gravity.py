import numpy as np
from .base import AbstractForce
from ..core.state import SystemState


class UniformGravity(AbstractForce):
    """
    Uniform gravitational field: F_i = m_i * g_vec.

    Parameters
    ----------
    g         : float — magnitude in m/s^2 (default 9.81)
    direction : array-like shape (D,) — unit vector for gravity direction.
                Default [0, -1] means downward in 2D.
    """

    def __init__(self, g: float = 9.81, direction: list[float] | None = None):
        self.g = g
        self.direction = np.array(direction if direction is not None else [0.0, -1.0])

    def compute(self, state: SystemState) -> np.ndarray:
        g_vec = self.g * self.direction                             # (D,)
        return state.masses[:, np.newaxis] * g_vec[np.newaxis, :]  # (N, D)

    def potential_energy(self, state: SystemState) -> float:
        # U = -m . g_vec . r  summed over all particles
        heights = state.positions @ self.direction                  # (N,)
        return float(-self.g * np.dot(state.masses, heights))


class NBodyGravity(AbstractForce):
    """
    N-body gravitational attraction: F_ij = G * m_i * m_j / r_ij^2 * r_hat_ij.

    Fully vectorized. Softening epsilon avoids the singularity at r -> 0.

    Parameters
    ----------
    G       : float — gravitational constant.
                      Use 6.674e-11 for SI, 1.0 for normalized units.
    epsilon : float — softening length (default 1e-5)
    """

    def __init__(self, G: float = 6.674e-11, epsilon: float = 1e-5):
        self.G = G
        self.epsilon = epsilon

    def compute(self, state: SystemState) -> np.ndarray:
        pos = state.positions   # (N, D)
        m   = state.masses      # (N,)

        # dr[i,j] = pos[j] - pos[i], shape (N, N, D)
        dr = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]

        # Squared distances + softening, shape (N, N)
        r2 = np.einsum('ijk,ijk->ij', dr, dr) + self.epsilon ** 2

        # |r|^3 for force magnitude; avoid self-interaction on diagonal
        r3 = r2 ** 1.5
        np.fill_diagonal(r3, 1.0)

        # m_j / r_ij^3 broadcast over i, shape (N, N)
        mj_over_r3 = m[np.newaxis, :] / r3

        # F_i = G * m_i * sum_j( m_j * dr_ij / |r_ij|^3 )
        forces = self.G * m[:, np.newaxis] * np.einsum('ij,ijk->ik', mj_over_r3, dr)
        return forces  # (N, D)

    def potential_energy(self, state: SystemState) -> float:
        pos = state.positions
        m   = state.masses

        dr = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]
        r2 = np.einsum('ijk,ijk->ij', dr, dr) + self.epsilon ** 2
        r  = np.sqrt(r2)
        np.fill_diagonal(r, np.inf)

        mi_mj = m[:, np.newaxis] * m[np.newaxis, :]
        return float(-0.5 * self.G * np.sum(mi_mj / r))

