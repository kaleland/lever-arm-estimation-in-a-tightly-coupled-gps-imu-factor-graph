
import numpy as np
from states import State
from util.so3 import expmap as so3_expmap
from numpy import ndarray
def reorthogonalize_so3(R: np.ndarray) -> np.ndarray:
    """Re-orthogonalize a 3x3 matrix to ensure it is a valid rotation matrix (SO(3)).

    Parameters
    ----------
    R : np.ndarray
        A 3x3 matrix that may have drifted from SO(3).

    Returns
    -------
    R_ortho : np.ndarray
        The closest valid rotation matrix to R in the Frobenius norm sense.

    """
    # Compute the SVD
    U, _, Vt = np.linalg.svd(R)
    
    # Ensure a proper rotation (det = +1)
    R_ortho = U @ Vt
    if np.linalg.det(R_ortho) < 0:
        # Flip the sign of the last column of U
        U[:, -1] *= -1
        R_ortho = U @ Vt
    
    return R_ortho

class SO3State(State):

    def __init__(self, time: float, value: ndarray):
        """Initialize."""
        delta_dimensions = 3
        if value.shape != (3, 3):
            raise ValueError("SO3State requires a 3x3 matrix.")
        super().__init__(time, value, delta_dimensions)

    @State.value.setter
    def value(self, value: ndarray):
        
        self._value = value.copy()

    @State.proposed_value.setter
    def proposed_value(self, proposed_value: ndarray) -> ndarray:
        
        self._proposed_value = proposed_value.copy() if proposed_value is not None else None

    def update(self, delta: ndarray, proposed: bool = False):
        
        if proposed:
            self.proposed_value = self.value @ so3_expmap(delta)
            self.proposed_value = reorthogonalize_so3(self.proposed_value)
        else:
            self.old_value = self.value.copy()
            self.value = self.value @ so3_expmap(delta)
            self.value = reorthogonalize_so3(self.value)

    def accept_proposed_value(self):
        
        if self.proposed_value is None:
            raise ValueError("No proposed value to accept.")
        self.value = self.proposed_value.copy()
        self.proposed_value = None

    def revert(self):
        """Revert to the old value."""
        if self.old_value is not None:
            self.value = self.old_value.copy()
            self.old_value = None
        else:
            print('WARNING: No old value to revert to.')