from __future__ import annotations
import numpy as np
from ..core.state import SystemState
from ..forces.base import AbstractForce


class DataRecorder:
    """
    Accumulates trajectory data during a simulation run.

    Uses Python lists as append buffers during the run, then converts to
    NumPy arrays on demand via properties. This avoids costly array
    resizing mid-simulation.

    Parameters
    ----------
    record_every  : int  — record every Nth outer step (default 1 = record all)
    record_energy : bool — whether to compute and store kinetic/potential energy
    """

    def __init__(self, record_every: int = 1, record_energy: bool = True):
        self.record_every  = record_every
        self.record_energy = record_energy

        self._step_counter: int = 0
        self._times:      list[float]      = []
        self._positions:  list[np.ndarray] = []
        self._velocities: list[np.ndarray] = []
        self._ke:         list[float]      = []
        self._pe:         list[float]      = []

    def record(self, state: SystemState, forces: list[AbstractForce]) -> None:
        """Called by Simulation.step() once per outer timestep."""
        if self._step_counter % self.record_every == 0:
            self._times.append(state.time)
            self._positions.append(state.positions.copy())
            self._velocities.append(state.velocities.copy())
            if self.record_energy:
                ke = state.kinetic_energy()
                pe = sum(f.potential_energy(state) for f in forces)
                self._ke.append(ke)
                self._pe.append(pe)
        self._step_counter += 1

    def clear(self) -> None:
        """Reset all recorded data."""
        self._times.clear()
        self._positions.clear()
        self._velocities.clear()
        self._ke.clear()
        self._pe.clear()
        self._step_counter = 0

    # --- Data accessors (lazy conversion to NumPy) ------------------

    @property
    def times(self) -> np.ndarray:
        """Shape: (T,)"""
        return np.array(self._times)

    @property
    def positions(self) -> np.ndarray:
        """Shape: (T, N, D)"""
        return np.stack(self._positions, axis=0)

    @property
    def velocities(self) -> np.ndarray:
        """Shape: (T, N, D)"""
        return np.stack(self._velocities, axis=0)

    @property
    def kinetic_energy(self) -> np.ndarray:
        """Shape: (T,)"""
        return np.array(self._ke)

    @property
    def potential_energy(self) -> np.ndarray:
        """Shape: (T,)"""
        return np.array(self._pe)

    @property
    def total_energy(self) -> np.ndarray:
        """Shape: (T,)"""
        return self.kinetic_energy + self.potential_energy

    def __len__(self) -> int:
        return len(self._times)

    def __repr__(self) -> str:
        return (
            f"DataRecorder(steps_recorded={len(self)}, "
            f"record_every={self.record_every}, "
            f"record_energy={self.record_energy})"
        )
