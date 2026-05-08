
from states import VectorState
from numpy import ndarray

class R3State(VectorState):

    def __init__(self, time: float, value: ndarray):
        """Initialize."""
        delta_dimensions = 3
        if value.shape != (delta_dimensions,):
            raise ValueError("R3State requires a 3-dimensional vector.")
        super().__init__(time, value, delta_dimensions)