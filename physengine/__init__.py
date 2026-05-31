"""
physengine — Modular particle simulation engine for scientific research.

Quick-start
-----------
    from physengine import ParticleSystem, Simulation
    from physengine.forces import UniformGravity, LinearDrag
    from physengine.integrators import get_integrator
    from physengine.recording import DataRecorder
    from physengine.visualization import RealtimeAnimator

    ps = ParticleSystem(dim=2)
    ps.add(mass=1.0, position=[0.0, 5.0], velocity=[2.0, 0.0])
    state = ps.build()

    recorder   = DataRecorder()
    integrator = get_integrator("verlet")

    sim = Simulation(state=state, integrator=integrator, dt=0.01, recorder=recorder)
    sim.add_force(UniformGravity(g=9.81))
    sim.add_force(LinearDrag(b=0.1))

    anim = RealtimeAnimator(sim, xlim=(-2, 12), ylim=(-2, 8), trail_length=100)
    anim.run()
"""

from .core.state      import SystemState
from .core.particle   import ParticleSystem
from .core.simulation import Simulation

from .forces.gravity        import UniformGravity, NBodyGravity
from .forces.spring         import SpringForce, MassSpringNetwork
from .forces.drag           import LinearDrag, QuadraticDrag
from .forces.field          import CustomForceField
from .forces.post_newtonian import PostNewtonianForce

from .integrators     import (
    get_integrator,
    RK4Integrator,
    VerletIntegrator,
    LeapfrogIntegrator,
)

from .recording.recorder  import DataRecorder
from .recording.exporters import NumpyExporter, CSVExporter, HDF5Exporter

from .visualization.animator import RealtimeAnimator
from .visualization.renderer import Renderer2D
from .visualization.renderer3d import Renderer3D

from .utils.orbital import (
    orbital_period, 
    vis_viva_velocity, 
    hohmann_transfer, 
    keplerian_to_cartesian
)

from .fluids import (
    SPHDensityComputer,
    SPHPressureForce,
    SPHViscosityForce,
    SPHSurfaceTension,
    BoxBoundaryForce,
    SPHParticleInitializer,
)

__version__ = "0.1.0"

__all__ = [
    # Core
    "SystemState", "ParticleSystem", "Simulation",
    # Forces
    "UniformGravity", "NBodyGravity",
    "SpringForce", "MassSpringNetwork",
    "LinearDrag", "QuadraticDrag",
    "CustomForceField",
    "PostNewtonianForce",
    # Integrators
    "get_integrator", "RK4Integrator", "VerletIntegrator", "LeapfrogIntegrator",
    # Recording
    "DataRecorder", "NumpyExporter", "CSVExporter", "HDF5Exporter",
    # Visualization
    "RealtimeAnimator", "Renderer2D", "Renderer3D",
    # Orbital
    "orbital_period", "vis_viva_velocity", "hohmann_transfer", "keplerian_to_cartesian",
    # Fluids / SPH
    "SPHDensityComputer", "SPHPressureForce", "SPHViscosityForce", "SPHSurfaceTension",
    "BoxBoundaryForce", "SPHParticleInitializer",
]
