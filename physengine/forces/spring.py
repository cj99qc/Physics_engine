import numpy as np
from .base import AbstractForce
from ..core.state import SystemState


class SpringForce(AbstractForce):
    """
    Hooke's law spring between two particles i and j.
    F = k * (|r_ij| - L0) * r_hat_ij

    Parameters
    ----------
    i, j : int   — particle indices
    k    : float — spring constant (N/m)
    L0   : float — rest length (m)
    """

    def __init__(self, i: int, j: int, k: float, L0: float):
        self.i  = i
        self.j  = j
        self.k  = k
        self.L0 = L0

    def compute(self, state: SystemState) -> np.ndarray:
        forces = np.zeros_like(state.positions)
        dr   = state.positions[self.j] - state.positions[self.i]
        dist = float(np.linalg.norm(dr))
        if dist < 1e-12:
            return forces
        r_hat = dr / dist
        f_mag = self.k * (dist - self.L0)
        forces[self.i] += f_mag * r_hat
        forces[self.j] -= f_mag * r_hat
        return forces

    def potential_energy(self, state: SystemState) -> float:
        dr   = state.positions[self.j] - state.positions[self.i]
        dist = float(np.linalg.norm(dr))
        return 0.5 * self.k * (dist - self.L0) ** 2


class MassSpringNetwork(AbstractForce):
    """
    Efficient batched spring network for cloth, jelly, or chains.
    Stores all connections as arrays for fully vectorized computation.

    Parameters
    ----------
    edges : (E, 2) int array  — particle index pairs
    ks    : (E,) float array  — spring constants
    L0s   : (E,) float array  — rest lengths
    """

    def __init__(
        self,
        edges: np.ndarray,
        ks:    np.ndarray,
        L0s:   np.ndarray,
    ):
        self.edges = np.asarray(edges, dtype=np.int32)    # (E, 2)
        self.ks    = np.asarray(ks,    dtype=np.float64)  # (E,)
        self.L0s   = np.asarray(L0s,   dtype=np.float64)  # (E,)

    @classmethod
    def from_grid(
        cls,
        nx: int,
        ny: int,
        k_structural: float,
        k_shear: float,
        spacing: float,
    ) -> "MassSpringNetwork":
        """
        Build a cloth-like spring network from an nx x ny grid.
        Creates structural (axis-aligned) and shear (diagonal) springs.
        Particle index = i * ny + j.
        """
        edges: list[list[int]] = []
        ks:    list[float]     = []
        L0s:   list[float]     = []

        def idx(i, j):
            return i * ny + j

        for i in range(nx):
            for j in range(ny):
                if i + 1 < nx:
                    edges.append([idx(i, j), idx(i + 1, j)])
                    ks.append(k_structural)
                    L0s.append(spacing)
                if j + 1 < ny:
                    edges.append([idx(i, j), idx(i, j + 1)])
                    ks.append(k_structural)
                    L0s.append(spacing)
                if i + 1 < nx and j + 1 < ny:
                    edges.append([idx(i, j), idx(i + 1, j + 1)])
                    ks.append(k_shear)
                    L0s.append(spacing * np.sqrt(2))
                    edges.append([idx(i + 1, j), idx(i, j + 1)])
                    ks.append(k_shear)
                    L0s.append(spacing * np.sqrt(2))

        return cls(np.array(edges), np.array(ks), np.array(L0s))

    def compute(self, state: SystemState) -> np.ndarray:
        forces = np.zeros_like(state.positions)
        i_idx = self.edges[:, 0]   # (E,)
        j_idx = self.edges[:, 1]   # (E,)

        dr   = state.positions[j_idx] - state.positions[i_idx]   # (E, D)
        dist = np.linalg.norm(dr, axis=1, keepdims=True)         # (E, 1)
        dist = np.maximum(dist, 1e-12)
        r_hat = dr / dist                                         # (E, D)

        f_mag = self.ks[:, np.newaxis] * (dist - self.L0s[:, np.newaxis])  # (E, D)
        f_vec = f_mag * r_hat                                                # (E, D)

        np.add.at(forces, i_idx, f_vec)
        np.add.at(forces, j_idx, -f_vec)
        return forces

    def potential_energy(self, state: SystemState) -> float:
        i_idx = self.edges[:, 0]
        j_idx = self.edges[:, 1]
        dr   = state.positions[j_idx] - state.positions[i_idx]
        dist = np.linalg.norm(dr, axis=1)
        return float(0.5 * np.sum(self.ks * (dist - self.L0s) ** 2))
