
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar, Optional

T = TypeVar('T')

class State(ABC, Generic[T]):

    def __init__(
            self,
            time: float,
            value: T,
            delta_dimensions: int
    ):
        """Initialize."""
        self.time = time
        self.value = value
        self.delta_dimensions = delta_dimensions
        self.proposed_value: Optional[T] = None
        self.old_value: Optional[T] = None

    @abstractmethod
    def update(self, delta: Any, proposed: bool = False):
        """Update the state given a delta value/vector."""
        pass

    @abstractmethod
    def accept_proposed_value(self):
        """Accept the proposed value and update the state."""
        pass

    def revert(self):
        """Revert to the old value."""
        if self.old_value is not None:
            self.value = self.old_value
            self.old_value = None
        else:
            print('WARNING: No old value to revert to.')

    @property
    def time(self):
        
        return self._time
    @time.setter
    def time(self, time: float):
        
        self._time = time

    @property
    def delta_dimensions(self):
        
        return self._delta_dimensions
    @delta_dimensions.setter
    def delta_dimensions(self, delta_dimensions: int):
        
        self._delta_dimensions = delta_dimensions

    @property
    def value(self) -> T:
        
        return self._value
    @value.setter
    @abstractmethod
    def value(self, value: T):
        
        pass

    @property
    def proposed_value(self) -> Optional[T]:
        
        return self._proposed_value
    @proposed_value.setter
    @abstractmethod
    def proposed_value(self, proposed_value: T):
        
        pass

    def __repr__(self):
        
        return f"{self.__class__.__name__}(time={self.time}, value={self.value}, delta_dimensions={self.delta_dimensions}, proposed_value={self.proposed_value})"




