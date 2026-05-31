from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


@dataclass
class SystemState:
    """
    Snapshot of the entire particle system at one instant.

    Shapes (all float64):
        positions  : (N, D)  — D is spatial dimension (default 2)
        velocities : (N, D)
        masses     : (N,)
        time       : scalar float
    """
    positions:  np.ndarray          # (N, D)
    velocities: np.ndarray          # (N, D)
    masses:     np.ndarray          # (N,)
    time:       float = 0.0

    accelerations: np.ndarray | None = field(default=None, repr=False)

    @property
    def n_particles(self) -> int:
        return self.positions.shape[0]

    @property
    def dim(self) -> int:
        return self.positions.shape[1]

    def copy(self) -> "SystemState":
        return SystemState(
            positions=self.positions.copy(),
            velocities=self.velocities.copy(),
            masses=self.masses.copy(),
            time=self.time,
        )

    def kinetic_energy(self) -> float:
        """0.5 * sum(m_i * |v_i|^2)"""
        v2 = np.einsum('ij,ij->i', self.velocities, self.velocities)
        return float(0.5 * np.dot(self.masses, v2))
