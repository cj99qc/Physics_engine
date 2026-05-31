import numpy as np
from .base import AbstractIntegrator, AccelFn
from ..core.state import SystemState


class RK4Integrator(AbstractIntegrator):
    """
    Classic 4th-order Runge-Kutta integrator.

    Evaluates the acceleration function 4 times per step.
    High accuracy for smooth force fields and short-duration problems.
    Not symplectic — energy drifts slowly over very long runs.
    """

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

        def make(p, v, t_):
            return SystemState(positions=p, velocities=v, masses=m, time=t_)

        # k1 — derivatives at t
        a1 = accel_fn(make(pos, vel, t))
        k1_p, k1_v = vel, a1

        # k2 — derivatives at t + dt/2 using k1 estimate
        p2 = pos + 0.5 * dt * k1_p
        v2 = vel + 0.5 * dt * k1_v
        a2 = accel_fn(make(p2, v2, t + 0.5 * dt))
        k2_p, k2_v = v2, a2

        # k3 — derivatives at t + dt/2 using k2 estimate
        p3 = pos + 0.5 * dt * k2_p
        v3 = vel + 0.5 * dt * k2_v
        a3 = accel_fn(make(p3, v3, t + 0.5 * dt))
        k3_p, k3_v = v3, a3

        # k4 — derivatives at t + dt using k3 estimate
        p4 = pos + dt * k3_p
        v4 = vel + dt * k3_v
        a4 = accel_fn(make(p4, v4, t + dt))
        k4_p, k4_v = v4, a4

        new_pos = pos + (dt / 6.0) * (k1_p + 2*k2_p + 2*k3_p + k4_p)
        new_vel = vel + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)

        return SystemState(positions=new_pos, velocities=new_vel, masses=m, time=t + dt)
