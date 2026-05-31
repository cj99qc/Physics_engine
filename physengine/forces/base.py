from abc import ABC, abstractmethod
import numpy as np
from ..core.state import SystemState


class AbstractForce(ABC):
    """
    Base class for all force contributions.

    Every subclass must implement compute(). Optionally override
    potential_energy() to enable energy conservation diagnostics.
    """

    @abstractmethod
    def compute(self, state: SystemState) -> np.ndarray:
        """
        Return force vectors for all particles.

        Parameters
        ----------
        state : SystemState
            Current system state (treat as read-only — do not mutate).

        Returns
        -------
        np.ndarray, shape (N, D)
            Force on each particle in each spatial dimension.
        """
        ...

    def potential_energy(self, state: SystemState) -> float:
        """
        Potential energy of this force. Override for conservative forces.
        Dissipative forces (drag) should leave this at 0.0.
        """
        return 0.0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
