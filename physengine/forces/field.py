from __future__ import annotations
import numpy as np
from typing import Callable
from .base import AbstractForce
from ..core.state import SystemState

# Callable signature: (positions, velocities, masses, time) -> forces (N, D)
ForceCallable = Callable[[np.ndarray, np.ndarray, np.ndarray, float], np.ndarray]


class CustomForceField(AbstractForce):
    """
    Wrap any user-defined Python callable as a force.

    The callable must accept:
        positions  : np.ndarray (N, D)
        velocities : np.ndarray (N, D)
        masses     : np.ndarray (N,)
        time       : float

    And return:
        forces     : np.ndarray (N, D)

    Example
    -------
    def vortex(pos, vel, masses, t):
        return np.stack([-pos[:, 1], pos[:, 0]], axis=1) * 0.5

    sim.add_force(CustomForceField(vortex, name="vortex"))
    """

    def __init__(self, fn: ForceCallable, name: str = "custom"):
        self._fn   = fn
        self._name = name

    def compute(self, state: SystemState) -> np.ndarray:
        return np.asarray(
            self._fn(state.positions, state.velocities, state.masses, state.time),
            dtype=np.float64,
        )

    def __repr__(self) -> str:
        return f"CustomForceField(name={self._name!r})"
