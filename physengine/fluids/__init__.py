"""
physengine.fluids — Smoothed Particle Hydrodynamics (SPH) and fluid mechanics.

This sub-package implements the Navier-Stokes equations in Lagrangian
(particle) form using SPH discretization:

    Dv/Dt = -(1/rho) grad(P) + nu * laplacian(v) + g

Components:
    kernels       : SPH interpolation kernels (cubic spline, Wendland C2)
    sph_forces    : Pressure, viscosity, surface tension forces
    boundaries    : Domain boundary handling and particle initialization
"""

from .kernels import (
    cubic_spline_kernel,
    cubic_spline_gradient_factor,
    wendland_c2_kernel,
    wendland_c2_gradient_factor,
)
from .sph_forces import (
    SPHDensityComputer,
    SPHPressureForce,
    SPHViscosityForce,
    SPHSurfaceTension,
)
from .boundaries import BoxBoundaryForce, SPHParticleInitializer

__all__ = [
    # Kernels
    "cubic_spline_kernel",
    "cubic_spline_gradient_factor",
    "wendland_c2_kernel",
    "wendland_c2_gradient_factor",
    # SPH Forces
    "SPHDensityComputer",
    "SPHPressureForce",
    "SPHViscosityForce",
    "SPHSurfaceTension",
    # Boundaries
    "BoxBoundaryForce",
    "SPHParticleInitializer",
]
