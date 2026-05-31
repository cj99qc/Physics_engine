import numpy as np
from .base import AbstractForce
from ..core.state import SystemState


class PostNewtonianForce(AbstractForce):
    """
    First post-Newtonian (1PN) correction to Newtonian N-body gravity.

    Implements the Einstein-Infeld-Hoffmann (EIH) equations in harmonic gauge
    (GR, gamma=beta=1).  Must be added ALONGSIDE NBodyGravity -- this supplies
    only the correction, not the full gravitational acceleration.

    For a test particle in a Schwarzschild field (v_source=0, m_test->0) the
    formula reduces to:

        a^1PN = (GM/r^2 c^2) [(v^2 - 4GM/r) n_ij  -  4 (n_ij.v) v]

    where n_ij = (x_j - x_i)/r points FROM the test particle TOWARD the source.
    This reproduces the correct prograde perihelion precession:
        delta_phi = 6 pi G M / (a (1-e^2) c^2)  per orbit.

    Full EIH equations (harmonic gauge, Will 1993 eq. 6.82, translated to the
    convention n_ij = (x_j - x_i)/r pointing from i toward j):

        a_i^1PN = (1/c^2) sum_{j!=i} (G m_j / r_ij^2) {
            n_ij  [+v_i^2 + 2 v_j^2 - 4(v_i.v_j) - (3/2)(n_ij.v_j)^2
                   - 5 G m_i/r_ij - 4 G m_j/r_ij]
            + (v_i - v_j) [-4(v_i.n_ij) + 3(v_j.n_ij)]
        }
        + (7/2 c^2) sum_{j!=i} (G m_j / r_ij) a_j^N

    Sign note: Will uses n pointing OUTWARD from the source; here n_ij points
    INWARD (toward the source), so both brackets are sign-flipped relative to
    Will's eq. 6.82a.

    Parameters
    ----------
    G                : float -- gravitational constant (must match NBodyGravity)
    c                : float -- speed of light in the same unit system
    epsilon          : float -- pairwise softening length
    include_indirect : bool  -- include the (7/2 c^2) indirect acceleration term
                               (coupling through Newtonian accelerations).
                               Default True.
    """

    def __init__(
        self,
        G: float = 6.674e-11,
        c: float = 2.998e8,
        epsilon: float = 1e-5,
        include_indirect: bool = True,
    ):
        self.G = G
        self.c = c
        self.epsilon = epsilon
        self.include_indirect = include_indirect

    def compute(self, state: SystemState) -> np.ndarray:
        pos = state.positions    # (N, D)
        vel = state.velocities   # (N, D)
        m   = state.masses       # (N,)
        G, c2, eps = self.G, self.c ** 2, self.epsilon

        # Pairwise geometry
        # dr[i,j] = pos[j] - pos[i]  so n[i,j] points from i toward j
        dr = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]        # (N, N, D)
        r  = np.sqrt(np.einsum('ijk,ijk->ij', dr, dr) + eps ** 2) # (N, N)
        np.fill_diagonal(r, np.inf)
        n  = dr / r[:, :, np.newaxis]                              # (N, N, D)

        # Velocity scalars
        v2        = np.einsum('id,id->i', vel, vel)    # (N,)   |v_i|^2
        vi2       = v2[:, np.newaxis]                  # (N,1)
        vj2       = v2[np.newaxis, :]                  # (1,N)
        vi_dot_vj = vel @ vel.T                        # (N, N)  v_i . v_j
        n_dot_vi  = np.einsum('ijk,ik->ij', n, vel)   # (N, N)  n_ij . v_i
        n_dot_vj  = np.einsum('ijk,jk->ij', n, vel)   # (N, N)  n_ij . v_j
        dv        = vel[:, np.newaxis, :] - vel[np.newaxis, :, :] # (N, N, D)

        # Potential scalars
        Gm_over_r_i = G * m[:, np.newaxis] / r        # G m_i / r_ij  (N, N)
        Gm_over_r_j = G * m[np.newaxis, :] / r        # G m_j / r_ij  (N, N)

        # Newtonian accelerations (needed for the indirect 7/2 term)
        w_N = G * m[np.newaxis, :] / r ** 2           # G m_j / r_ij^2  (N, N)
        np.fill_diagonal(w_N, 0.0)
        a_newton = np.einsum('ij,ijk->ik', w_N, n)    # (N, D)

        # EIH bracket for the n_ij direction
        # (signs flipped vs Will because n_ij points toward source, not away)
        bracket_n = (
            + vi2
            + 2.0 * vj2
            - 4.0 * vi_dot_vj
            - 1.5 * n_dot_vj ** 2
            - 5.0 * Gm_over_r_i
            - 4.0 * Gm_over_r_j
        )   # (N, N)

        # EIH bracket for the (v_i - v_j) direction
        bracket_dv = -4.0 * n_dot_vi + 3.0 * n_dot_vj  # (N, N)

        # Weight: G m_j / (r_ij^2 c^2)
        weight = Gm_over_r_j / (r * c2)               # (N, N)
        np.fill_diagonal(weight, 0.0)

        # Direct 1PN acceleration
        a_1pn = (
            np.einsum('ij,ijk->ik', weight * bracket_n,  n)
            + np.einsum('ij,ijk->ik', weight * bracket_dv, dv)
        )   # (N, D)

        # Indirect 1PN term: (7/2 c^2) sum_j (G m_j / r_ij) a_j^N
        if self.include_indirect:
            w_ind = G * m[np.newaxis, :] / (r * c2)   # (N, N)
            np.fill_diagonal(w_ind, 0.0)
            a_1pn = a_1pn + 3.5 * np.einsum('ij,jk->ik', w_ind, a_newton)

        # Return forces: F_i = m_i * a_i^1PN
        return m[:, np.newaxis] * a_1pn

    def __repr__(self) -> str:
        return (
            f"PostNewtonianForce(G={self.G}, c={self.c}, "
            f"include_indirect={self.include_indirect})"
        )
