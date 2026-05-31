from __future__ import annotations
import numpy as np
from .state import SystemState
from ..forces.base import AbstractForce
from ..integrators.base import AbstractIntegrator


class Simulation:
    """
    Central orchestrator for a particle simulation.

    Parameters
    ----------
    state      : Initial SystemState
    integrator : Any AbstractIntegrator implementation
    dt         : Fixed outer timestep (seconds)
    substeps   : Sub-steps per outer step — each sub-step uses dt/substeps.
                 Useful for stiff springs without increasing recorder frequency.
    recorder   : Optional DataRecorder; if None, no data is saved.
    """

    def __init__(
        self,
        state:      SystemState,
        integrator: AbstractIntegrator,
        dt:         float = 0.01,
        substeps:   int = 1,
        recorder=None,
    ):
        self.state      = state
        self.integrator = integrator
        self.dt         = dt
        self.substeps   = substeps
        self.recorder   = recorder

        self._forces:     list[AbstractForce] = []
        self._step_count: int = 0
        self._events:     list[dict] = []

    # --- Force management -------------------------------------------

    def add_force(self, force: AbstractForce) -> "Simulation":
        """Add a force to the simulation. Returns self for chaining."""
        self._forces.append(force)
        return self

    def schedule_event(self, time: float, callback) -> "Simulation":
        """Schedules an arbitrary callback function to execute at a specific time."""
        self._events.append({
            "time": time,
            "callback": callback
        })
        self._events.sort(key=lambda e: e["time"])
        return self

    def remove_force(self, force: AbstractForce) -> None:
        self._forces.remove(force)

    def list_forces(self) -> list[AbstractForce]:
        return list(self._forces)

    # --- Acceleration computation -----------------------------------

    def compute_accelerations(self, state: SystemState) -> np.ndarray:
        """
        Sum all force contributions and divide by mass.
        Returns ndarray of shape (N, D).
        This is the accel_fn passed to integrators.
        """
        total = np.zeros_like(state.positions)
        for force in self._forces:
            total += force.compute(state)
        return total / state.masses[:, np.newaxis]

    # --- Simulation loop --------------------------------------------

    def step(self) -> None:
        """Advance by one outer timestep (dt), using substeps internally."""
        sub_dt = self.dt / self.substeps

        # Check and apply scheduled events for this timestep
        next_time = self.state.time + self.dt
        while self._events and self.state.time <= self._events[0]["time"] < next_time:
            e = self._events.pop(0)
            e["callback"](self)
            
            # Clear Leapfrog velocity cache in case the callback impulsively altered velocities
            if hasattr(self.integrator, '_vel_half'):
                self.integrator._vel_half = None

        for _ in range(self.substeps):
            self.state = self.integrator.step(
                state=self.state,
                accel_fn=self.compute_accelerations,
                dt=sub_dt,
            )
        self._step_count += 1
        if self.recorder is not None:
            self.recorder.record(self.state, self._forces)

    def run(self, n_steps: int) -> None:
        """Run for a fixed number of outer steps."""
        for _ in range(n_steps):
            self.step()

    def run_until(self, end_time: float) -> None:
        """Run until simulation time reaches end_time."""
        while self.state.time < end_time:
            self.step()

    @property
    def time(self) -> float:
        return self.state.time

    @property
    def step_count(self) -> int:
        return self._step_count
