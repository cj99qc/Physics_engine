from .rk4 import RK4Integrator
from .verlet import VerletIntegrator, LeapfrogIntegrator
from .base import AbstractIntegrator, AccelFn

_REGISTRY = {
    "rk4":      RK4Integrator,
    "verlet":   VerletIntegrator,
    "leapfrog": LeapfrogIntegrator,
}


def get_integrator(name: str) -> AbstractIntegrator:
    """
    Factory for runtime integrator selection.

    Parameters
    ----------
    name : str
        One of "rk4", "verlet", or "leapfrog".

    Returns
    -------
    AbstractIntegrator instance ready to use.
    """
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown integrator '{name}'. Choose from: {list(_REGISTRY)}")
    return _REGISTRY[key]()


__all__ = [
    "AbstractIntegrator", "AccelFn",
    "RK4Integrator", "VerletIntegrator", "LeapfrogIntegrator",
    "get_integrator",
]
