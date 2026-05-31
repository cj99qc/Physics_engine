import numpy as np


def orbital_period(a: float, mu: float) -> float:
    """Returns the time T of one full orbit for semi-major axis a."""
    return 2.0 * np.pi * np.sqrt(a ** 3 / mu)


def vis_viva_velocity(r: float, a: float, mu: float) -> float:
    """Computes orbital velocity magnitude at a given radius."""
    return np.sqrt(mu * (2.0 / r - 1.0 / a))


def hohmann_transfer(r1: float, r2: float, mu: float) -> tuple[float, float, float]:
    """
    Returns (dv1, dv2, time_of_flight) for a Hohmann transfer from a circular 
    orbit of radius r1 to a circular orbit of radius r2.
    """
    v1 = np.sqrt(mu / r1)
    v2 = np.sqrt(mu / r2)
    a_transfer = 0.5 * (r1 + r2)
    
    v_trans1 = np.sqrt(mu * (2.0 / r1 - 1.0 / a_transfer))
    v_trans2 = np.sqrt(mu * (2.0 / r2 - 1.0 / a_transfer))
    
    dv1 = v_trans1 - v1
    dv2 = v2 - v_trans2
    tof = np.pi * np.sqrt(a_transfer ** 3 / mu)
    
    return dv1, dv2, tof


def keplerian_to_cartesian(
    a: float, 
    e: float, 
    i: float, 
    omega: float, 
    Omega: float, 
    nu: float, 
    mu: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts Keplerian orbital elements to Cartesian state vectors.
    All angular elements (i, omega, Omega, nu) must be in radians.
    
    Uses standard astrodynamics formulas mapping through the PQW perifocal frame.
    Returns:
        pos_ijk (ndarray): 3D position vector
        vel_ijk (ndarray): 3D velocity vector
    """
    # Semi-latus rectum
    p = a * (1.0 - e ** 2)
    
    # Radius magnitude
    r = p / (1.0 + e * np.cos(nu))
    
    # Position and velocity in orbital plane (PQW frame)
    pos_pqw = np.array([r * np.cos(nu), r * np.sin(nu), 0.0])
    vel_pqw = np.sqrt(mu / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])
    
    # Rotation matrices from PQW to IJK frame
    cO, sO = np.cos(Omega), np.sin(Omega)
    co, so = np.cos(omega), np.sin(omega)
    ci, si = np.cos(i), np.sin(i)
    
    rot_O = np.array([[cO, -sO, 0.0], [sO, cO, 0.0], [0.0, 0.0, 1.0]])
    rot_i = np.array([[1.0, 0.0, 0.0], [0.0, ci, -si], [0.0, si, ci]])
    rot_o = np.array([[co, -so, 0.0], [so, co, 0.0], [0.0, 0.0, 1.0]])
    
    R = rot_O @ rot_i @ rot_o
    
    pos_ijk = R @ pos_pqw
    vel_ijk = R @ vel_pqw
    
    return pos_ijk, vel_ijk
