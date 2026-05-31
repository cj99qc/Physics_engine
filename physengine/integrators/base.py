from abc import ABC, abstractmethod
from typing import Callable
import numpy as np
from ..core.state import SystemState

# Callable signature: (SystemState) -> accelerations (N, D)
AccelFn = Callable[[SystemState], np.ndarray]


class AbstractIntegrator(ABC):
    """
    Strategy interface for numerical integration.

    Integrators receive a state, an acceleration function, and a timestep.
    They must return a NEW SystemState advanced by dt without mutating the input.
    """

    @abstractmethod
    def step(
        self,
        state:    SystemState,
        accel_fn: AccelFn,
        dt:       float,
    ) -> SystemState:
        """Advance state by dt and return the new SystemState."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
