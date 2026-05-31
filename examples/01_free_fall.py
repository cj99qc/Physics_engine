"""
Example 01: Free fall under uniform gravity with linear drag.

Runs headlessly (no visualization), records trajectory, exports to CSV,
and prints energy drift as a quick validation of the integrator.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from physengine import (
    ParticleSystem, Simulation, DataRecorder,
    UniformGravity, LinearDrag,
    VerletIntegrator, CSVExporter,
)

# Build system: single ball thrown sideways
ps = ParticleSystem(dim=2)
ps.add(mass=1.0, position=[0.0, 10.0], velocity=[3.0, 0.0], tag="ball")
state = ps.build()

recorder   = DataRecorder(record_every=1, record_energy=True)
integrator = VerletIntegrator()

sim = Simulation(state=state, integrator=integrator, dt=0.01, recorder=recorder)
sim.add_force(UniformGravity(g=9.81))
sim.add_force(LinearDrag(b=0.1))

sim.run(n_steps=1000)  # 10 simulated seconds

# Export
os.makedirs("output", exist_ok=True)
CSVExporter("output/01_trajectory.csv", "output/01_energy.csv").export(recorder)

e = recorder.total_energy
print(f"Simulated time   : {sim.time:.2f} s")
print(f"Final position   : {sim.state.positions[0]}")
print(f"Final velocity   : {sim.state.velocities[0]}")
print(f"Total E at t=0   : {e[0]:.4f} J")
print(f"Total E at t=end : {e[-1]:.4f} J")
print(f"Energy removed by drag: {e[0] - e[-1]:.4f} J  (positive = dissipated)")
