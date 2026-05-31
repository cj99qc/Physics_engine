import numpy as np
from .base import AbstractIntegrator, AccelFn
from ..core.state import SystemState


class VerletIntegrator(AbstractIntegrator):
    """
    Velocity Verlet integrator.

    Symplectic (phase-space area-preserving) — superior long-term energy
    conservation vs RK4 for conservative systems.
    Requires only 1 force evaluation per step by caching the previous acceleration.

    Reset _prev_accel = None to force recomputation on the next step.
    """

    def __init__(self):
        self._prev_accel: np.ndarray | None = None

    def step(
        self,
        state:    SystemState,
        accel_fn: AccelFn,
        dt:       float,
    ) -> SystemState:
        pos = state.positions
        vel = state.velocities
        m   = state.masses
        t   = state.time

        # a(t): use cached value if available
        a_t = self._prev_accel if self._prev_accel is not None else accel_fn(state)

        # x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2
        new_pos = pos + vel * dt + 0.5 * a_t * dt ** 2

        # a(t+dt)
        next_temp = SystemState(positions=new_pos, velocities=vel, masses=m, time=t + dt)
        a_t1 = accel_fn(next_temp)

        # v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
        new_vel = vel + 0.5 * (a_t + a_t1) * dt

        self._prev_accel = a_t1

        return SystemState(positions=new_pos, velocities=new_vel, masses=m, time=t + dt)


class LeapfrogIntegrator(AbstractIntegrator):
    """
    Leapfrog (Kick-Drift-Kick) integrator.

    Fully symplectic and time-reversible. Velocity and position are staggered
    by half a timestep, giving excellent long-term stability.
    Preferred for molecular dynamics and long orbital simulations.

    The first step bootstraps the half-step velocity from v(0) and a(0).
    """

    def __init__(self):
        self._vel_half: np.ndarray | None = None

    def step(
        self,
        state:    SystemState,
        accel_fn: AccelFn,
        dt:       float,
    ) -> SystemState:
        pos = state.positions
        vel = state.velocities
        m   = state.masses
        t   = state.time

        a = accel_fn(state)

        # Bootstrap: compute v(t + dt/2) on the first step
        if self._vel_half is None:
            self._vel_half = vel + 0.5 * dt * a

        # Drift: x(t+dt) = x(t) + v(t+dt/2) * dt
        new_pos = pos + self._vel_half * dt

        # Kick: v(t+3dt/2) = v(t+dt/2) + a(t+dt)*dt
        next_temp = SystemState(positions=new_pos, velocities=self._vel_half, masses=m, time=t + dt)
        a_new = accel_fn(next_temp)
        self._vel_half = self._vel_half + a_new * dt

        # Synchronised velocity v(t+dt) for output and energy calculation
        new_vel = self._vel_half - 0.5 * dt * a_new

        return SystemState(positions=new_pos, velocities=new_vel, masses=m, time=t + dt)
