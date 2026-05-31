import numpy as np
from .base import AbstractForce
from ..core.state import SystemState


class LinearDrag(AbstractForce):
    """
    Linear (viscous) drag: F = -b * v.
    Models Stokes drag in a viscous fluid.

    Parameters
    ----------
    b : float — drag coefficient (default 0.1)
    """

    def __init__(self, b: float = 0.1):
        self.b = b

    def compute(self, state: SystemState) -> np.ndarray:
        return -self.b * state.velocities


class QuadraticDrag(AbstractForce):
    """
    Quadratic (aerodynamic) drag: F = -c * |v| * v.
    Models air resistance at higher velocities.

    Parameters
    ----------
    c : float — drag coefficient (default 0.01)
    """

    def __init__(self, c: float = 0.01):
        self.c = c

    def compute(self, state: SystemState) -> np.ndarray:
        speeds = np.linalg.norm(state.velocities, axis=1, keepdims=True)  # (N, 1)
        return -self.c * speeds * state.velocities
