"""
Example 02: Spring pendulum with real-time visualization.

A very heavy anchor particle is connected to a light bob by a spring.
Gravity pulls the bob, creating pendulum-like oscillation.
Demonstrates RK4 integrator selection by string key.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from physengine import (
    ParticleSystem, Simulation, DataRecorder,
    UniformGravity, SpringForce,
    get_integrator, RealtimeAnimator,
)

ps = ParticleSystem(dim=2)
ps.add(mass=1e9, position=[0.0, 0.0], velocity=[0.0, 0.0], tag="anchor")
ps.add(mass=1.0, position=[0.0, -3.0], velocity=[1.5, 0.0], tag="bob")
state = ps.build()

recorder   = DataRecorder(record_every=5, record_energy=True)
integrator = get_integrator("rk4")  # select by name at runtime

sim = (
    Simulation(state=state, integrator=integrator, dt=0.005, recorder=recorder)
    .add_force(UniformGravity(g=9.81))
    .add_force(SpringForce(i=0, j=1, k=50.0, L0=3.0))
)

anim = RealtimeAnimator(
    simulation=sim,
    steps_per_frame=5,
    interval=16,
    xlim=(-5.0, 5.0),
    ylim=(-6.0, 2.0),
    trail_length=300,
    title="Spring Pendulum (RK4)",
)
anim.run()
