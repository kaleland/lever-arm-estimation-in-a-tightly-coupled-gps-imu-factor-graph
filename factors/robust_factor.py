from factors import Factor
from dataclasses import dataclass
from typing import Callable, Any

@dataclass
class RobustFactor(Factor):
    """A factor that includes robust error and Jacobian functions.
    
    Attributes:
        error_func: A function to compute the error.
        jacobian_func: A function to compute the Jacobians.

    """

    robust_weight_func: Callable[..., Any]