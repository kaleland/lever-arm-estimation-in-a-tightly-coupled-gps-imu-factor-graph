
from states import State

class FloatState(State[float]):

    def __init__(self, time: float, value: float, delta_dimensions: int = 1):
        """Initialize."""
        super().__init__(time, value, delta_dimensions)

    @State.value.setter
    def value(self, value: float):
        
        self._value = value

    @State.proposed_value.setter
    def proposed_value(self, proposed_value: float):
        
        self._proposed_value = proposed_value

    def update(self, delta: float, proposed: bool = False):
        
        if proposed:
            self.proposed_value = self.value + delta
        else:
            self.old_value = self.value+0.0
            self.value = self.value + delta


    def accept_proposed_value(self):
        
        self.value = self.proposed_value
        self.proposed_value = None