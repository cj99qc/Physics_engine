from __future__ import annotations
import numpy as np
from .state import SystemState


class ParticleSystem:
    """
    Builder / factory for SystemState.

    Example
    -------
    ps = ParticleSystem(dim=2)
    ps.add(mass=1.0, position=[0.0, 0.0], velocity=[1.0, 0.0])
    ps.add(mass=2.0, position=[3.0, 0.0])
    state = ps.build()
    """

    def __init__(self, dim: int = 2):
        self._dim = dim
        self._masses:     list[float]       = []
        self._positions:  list[list[float]] = []
        self._velocities: list[list[float]] = []
        self._tags:       list[str]         = []

    def add(
        self,
        mass: float,
        position:  list[float] | np.ndarray,
        velocity:  list[float] | np.ndarray | None = None,
        tag: str = "",
    ) -> "ParticleSystem":
        """Add a single particle. Returns self for chaining."""
        if velocity is None:
            velocity = [0.0] * self._dim
        self._masses.append(float(mass))
        self._positions.append([float(x) for x in position])
        self._velocities.append([float(x) for x in velocity])
        self._tags.append(tag)
        return self

    def add_grid(
        self,
        nx: int,
        ny: int,
        spacing: float,
        mass: float = 1.0,
        origin: tuple[float, float] = (0.0, 0.0),
    ) -> "ParticleSystem":
        """Add a regular nx x ny grid of particles (e.g. for cloth)."""
        for i in range(nx):
            for j in range(ny):
                pos = [origin[0] + i * spacing, origin[1] + j * spacing]
                self.add(mass=mass, position=pos, tag=f"grid_{i}_{j}")
        return self

    def build(self) -> SystemState:
        """Construct and return the immutable SystemState."""
        return SystemState(
            positions=np.array(self._positions,  dtype=np.float64),
            velocities=np.array(self._velocities, dtype=np.float64),
            masses=np.array(self._masses,         dtype=np.float64),
            time=0.0,
        )

    @property
    def n_particles(self) -> int:
        return len(self._masses)

    @property
    def tags(self) -> list[str]:
        return list(self._tags)
