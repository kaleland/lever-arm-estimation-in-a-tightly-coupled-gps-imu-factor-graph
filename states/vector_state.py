
from states import State
from numpy import ndarray

class VectorState(State[ndarray]):

    def __init__(self, time: float, value: ndarray, delta_dimensions: int):
        """Initialize."""
        super().__init__(time, value, delta_dimensions)

    @State.value.setter
    def value(self, value: ndarray):
        
        self._value = value.copy()

    @State.proposed_value.setter
    def proposed_value(self, proposed_value: ndarray):
        
        self._proposed_value = proposed_value.copy() if proposed_value is not None else None
 
    def update(self, delta: ndarray, proposed: bool = False):
        
        if proposed:
            self.proposed_value = self.value + delta
        else:
            self.old_value = self.value.copy()
            self.value = self.value + delta

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