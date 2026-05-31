"""
SPH (Smoothed Particle Hydrodynamics) forces for the Navier-Stokes equations.

The incompressible Navier-Stokes equations in Lagrangian form:

    Dv/Dt = -(1/rho) grad(P) + nu * laplacian(v) + g

SPH discretizes these as particle-particle interactions:

  1. Density:    rho_i = sum_j m_j W(|r_i - r_j|, h)
  2. Pressure:   P_i = k (rho_i - rho_0)            [Tait EOS]
  3. Pressure Force:  F_pressure = -m_i sum_j m_j (P_i/rho_i^2 + P_j/rho_j^2) grad W_ij
  4. Viscosity:       F_visc = mu * m_i sum_j m_j (v_j - v_i) / rho_j * laplacian W_ij
  5. External:        F_ext = m_i * g

This module provides AbstractForce subclasses that plug directly into
the existing Simulation loop.
"""
from __future__ import annotations
import numpy as np
from ..forces.base import AbstractForce
from ..core.state import SystemState
from .kernels import cubic_spline_kernel, cubic_spline_gradient_factor


class SPHDensityComputer:
    """
    Computes SPH density for all particles. Not a force — called by the
    SPH forces before they compute accelerations.

    Stores computed densities on the state object's accelerations field
    (repurposed) or in an external array.

    Parameters
    ----------
    h        : float — smoothing length
    rest_rho : float — rest density (kg/m^3)
    """

    def __init__(self, h: float = 0.1, rest_rho: float = 1000.0):
        self.h = h
        self.rest_rho = rest_rho
        self._densities: np.ndarray | None = None

    def compute_densities(self, state: SystemState) -> np.ndarray:
        """Compute SPH density for all particles. Returns (N,) array."""
        pos = state.positions       # (N, D)
        masses = state.masses       # (N,)
        N = pos.shape[0]
        dim = pos.shape[1]

        # Pairwise distances
        dr = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (N, N, D)
        r = np.sqrt(np.einsum('ijk,ijk->ij', dr, dr))       # (N, N)

        # Kernel values
        W = cubic_spline_kernel(r, self.h, dim)              # (N, N)

        # rho_i = sum_j m_j W_ij
        rho = np.einsum('j,ij->i', masses, W)                # (N,)

        # Clamp to avoid division by zero
        rho = np.maximum(rho, 1e-6)

        self._densities = rho
        return rho

    @property
    def densities(self) -> np.ndarray:
        if self._densities is None:
            raise RuntimeError("Call compute_densities() first.")
        return self._densities


class SPHPressureForce(AbstractForce):
    """
    SPH pressure gradient force using the Tait equation of state:
        P = k * (rho - rho_0)

    where k controls stiffness. Higher k = more incompressible fluid.

    The symmetric pressure gradient formulation ensures momentum conservation:
        F_i = -m_i * sum_j m_j (P_i/rho_i^2 + P_j/rho_j^2) grad_i W_ij

    Parameters
    ----------
    density_computer : SPHDensityComputer instance
    k                : pressure stiffness constant
    """

    def __init__(self, density_computer: SPHDensityComputer, k: float = 1000.0):
        self.dc = density_computer
        self.k = k

    def compute(self, state: SystemState) -> np.ndarray:
        pos = state.positions       # (N, D)
        masses = state.masses       # (N,)
        N, D = pos.shape

        rho = self.dc.compute_densities(state)  # (N,)
        P = self.k * (rho - self.dc.rest_rho)   # (N,) pressure

        # Pairwise vectors and distances
        dr = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (N, N, D)
        r = np.sqrt(np.einsum('ijk,ijk->ij', dr, dr))       # (N, N)

        # Gradient factor: dW/dr / r
        grad_factor = cubic_spline_gradient_factor(r, self.dc.h, D)  # (N, N)

        # Pressure term: P_i/rho_i^2 + P_j/rho_j^2
        P_over_rho2 = P / rho ** 2                           # (N,)
        pressure_term = P_over_rho2[:, np.newaxis] + P_over_rho2[np.newaxis, :]  # (N, N)

        # F_i = -m_i * sum_j m_j * pressure_term * grad_factor * dr_ij
        # grad_factor already has 1/r built in, so grad W = grad_factor * dr
        weight = masses[np.newaxis, :] * pressure_term * grad_factor  # (N, N)
        np.fill_diagonal(weight, 0.0)

        forces = -masses[:, np.newaxis] * np.einsum('ij,ijk->ik', weight, dr)  # (N, D)
        return forces

    def potential_energy(self, state: SystemState) -> float:
        return 0.0  # Pressure doesn't have a simple potential


class SPHViscosityForce(AbstractForce):
    """
    SPH artificial viscosity (Monaghan 1992) for modeling fluid viscosity.

    Uses the formulation:
        F_visc_i = mu * m_i * sum_j  m_j * (v_j - v_i) / rho_j * (1/|r_ij|) * dW/dr

    This approximates the Laplacian of velocity in the Navier-Stokes viscous term.

    Parameters
    ----------
    density_computer : SPHDensityComputer instance
    mu               : dynamic viscosity coefficient
    """

    def __init__(self, density_computer: SPHDensityComputer, mu: float = 0.1):
        self.dc = density_computer
        self.mu = mu

    def compute(self, state: SystemState) -> np.ndarray:
        pos = state.positions       # (N, D)
        vel = state.velocities      # (N, D)
        masses = state.masses       # (N,)
        N, D = pos.shape

        rho = self.dc.densities     # (N,) — must be precomputed by pressure force

        # Pairwise
        dr = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (N, N, D)
        dv = vel[np.newaxis, :, :] - vel[:, np.newaxis, :]  # (N, N, D) — v_j - v_i
        r = np.sqrt(np.einsum('ijk,ijk->ij', dr, dr))       # (N, N)

        # Gradient factor
        grad_factor = cubic_spline_gradient_factor(r, self.dc.h, D)  # (N, N)

        # v_ij dot r_ij / (|r_ij|^2 + epsilon)
        vr_dot = np.einsum('ijk,ijk->ij', dv, dr)            # (N, N)
        r2_eps = np.einsum('ijk,ijk->ij', dr, dr) + 0.01 * self.dc.h ** 2  # (N, N)

        # Viscous term: mu * m_j / rho_j * vr_dot / r2_eps * grad_factor
        visc_scalar = self.mu * masses[np.newaxis, :] / rho[np.newaxis, :] * vr_dot / r2_eps  # (N, N)
        np.fill_diagonal(visc_scalar, 0.0)

        # F_i = m_i * sum_j visc_scalar_ij * dr_ij * grad_factor_ij  ... but
        # Actually we use a simpler Morris (1997) viscosity:
        # F_i = mu * sum_j m_j (v_j - v_i) / rho_j * laplacian_W
        # where laplacian_W ≈ 2 * grad_factor * D (for D dimensions)
        weight = 2.0 * self.mu * masses[np.newaxis, :] / rho[np.newaxis, :] * grad_factor  # (N, N)
        np.fill_diagonal(weight, 0.0)

        forces = masses[:, np.newaxis] * np.einsum('ij,ijk->ik', weight, dv)  # (N, D)
        return forces

    def potential_energy(self, state: SystemState) -> float:
        return 0.0


class SPHSurfaceTension(AbstractForce):
    """
    Simple color-field based surface tension for SPH.

    Computes the normal field from the density gradient and applies
    a cohesion force toward the surface to create surface tension effects.

    Parameters
    ----------
    density_computer : SPHDensityComputer instance
    gamma            : surface tension coefficient
    """

    def __init__(self, density_computer: SPHDensityComputer, gamma: float = 0.01):
        self.dc = density_computer
        self.gamma = gamma

    def compute(self, state: SystemState) -> np.ndarray:
        pos = state.positions
        masses = state.masses
        N, D = pos.shape

        rho = self.dc.densities

        # Pairwise
        dr = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        r = np.sqrt(np.einsum('ijk,ijk->ij', dr, dr))

        grad_factor = cubic_spline_gradient_factor(r, self.dc.h, D)

        # Normal = gradient of color field (density / rest_density)
        color = rho / self.dc.rest_rho  # (N,)
        weight = masses[np.newaxis, :] / rho[np.newaxis, :] * grad_factor  # (N, N)
        np.fill_diagonal(weight, 0.0)

        # grad(color)_i = sum_j m_j/rho_j * color_j * grad_W_ij
        normal = np.einsum('ij,ijk->ik', weight * color[np.newaxis, :], dr)  # (N, D)

        # Surface tension force: F = -gamma * |normal| * normal_hat
        normal_mag = np.linalg.norm(normal, axis=1, keepdims=True)  # (N, 1)
        safe_mag = np.maximum(normal_mag, 1e-12)
        forces = -self.gamma * masses[:, np.newaxis] * normal_mag * (normal / safe_mag)

        return forces

    def potential_energy(self, state: SystemState) -> float:
        return 0.0
