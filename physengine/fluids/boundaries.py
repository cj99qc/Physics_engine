"""
Boundary handling for SPH simulations.

Fluid particles need to be contained within a domain. This module
provides boundary forces and helper utilities for common geometries.
"""
import numpy as np
from ..forces.base import AbstractForce
from ..core.state import SystemState


class BoxBoundaryForce(AbstractForce):
    """
    Repulsive boundary force that keeps SPH particles inside a rectangular box.

    Uses a Lennard-Jones-style repulsion from the walls. Particles approaching
    within a distance `margin` of any wall face a steep repulsive force
    pushing them back inward.

    Parameters
    ----------
    bounds : tuple of (min, max) for each dimension
             e.g. ((0, 10), (0, 10)) for a 2D 10x10 box
    stiffness : strength of the repulsive force
    margin    : distance from wall at which repulsion activates
    """

    def __init__(
        self,
        bounds: tuple[tuple[float, float], ...],
        stiffness: float = 10000.0,
        margin: float = 0.05,
    ):
        self.bounds = bounds
        self.stiffness = stiffness
        self.margin = margin

    def compute(self, state: SystemState) -> np.ndarray:
        pos = state.positions   # (N, D)
        forces = np.zeros_like(pos)

        for d, (lo, hi) in enumerate(self.bounds):
            # Distance from lower wall
            d_lo = pos[:, d] - lo
            mask_lo = d_lo < self.margin
            forces[mask_lo, d] += self.stiffness * (self.margin - d_lo[mask_lo]) ** 2

            # Distance from upper wall
            d_hi = hi - pos[:, d]
            mask_hi = d_hi < self.margin
            forces[mask_hi, d] -= self.stiffness * (self.margin - d_hi[mask_hi]) ** 2

        return forces

    def potential_energy(self, state: SystemState) -> float:
        return 0.0


class SPHParticleInitializer:
    """
    Utility to create uniformly-spaced SPH fluid particles
    filling a rectangular region.

    Parameters
    ----------
    region  : tuple of (min, max) per dimension
    spacing : distance between particles (sets resolution)
    mass    : mass per particle (computed from rho_0 * spacing^D if None)
    rho_0   : rest density for mass computation
    """

    @staticmethod
    def fill_box(
        region: tuple[tuple[float, float], ...],
        spacing: float,
        rho_0: float = 1000.0,
        mass: float | None = None,
    ) -> tuple[np.ndarray, float]:
        """
        Returns positions (N, D) and particle mass.
        """
        dim = len(region)

        # Generate grid points per dimension
        axes = []
        for lo, hi in region:
            ax = np.arange(lo + spacing / 2, hi, spacing)
            axes.append(ax)

        grid = np.meshgrid(*axes, indexing='ij')
        positions = np.column_stack([g.ravel() for g in grid])

        if mass is None:
            mass = rho_0 * spacing ** dim

        return positions, mass
