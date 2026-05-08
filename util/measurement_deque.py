
from collections import deque
import numpy as np
from typing import Optional

class TimedMeasurementQueue:

    def __init__(self, sort_idx: int = 0):
        """Initialize."""
        self.queue = deque()
        self.sort_idx = sort_idx

    def extend(self, measurements: np.ndarray):
        """Extend the queue with new measurements."""
        # for measurement in measurements:
        #     self.queue.append(measurement)
        self.queue.extend(measurements)

    def popleft_through(self, end_time: float) -> Optional[np.ndarray]:
        """Pop measurements from the queue until the end_time is reached.
        Returns the popped measurements as a NumPy array.
        """
        popped_measurements = []
        while self.queue and self.queue[0][self.sort_idx] <= end_time:
            popped_measurements.append(self.queue.popleft())
        
        if not popped_measurements:
            return None
        
        return np.array(popped_measurements)
    
    def popleft_before(self, end_time: float) -> Optional[np.ndarray]:
        """Pop measurements from the queue before the end_time.
        Returns the popped measurements as a NumPy array.
        """
        popped_measurements = []
        while self.queue and self.queue[0][self.sort_idx] < end_time:
            popped_measurements.append(self.queue.popleft())
        
        if not popped_measurements:
            return None
        
        return np.array(popped_measurements)
    
    
    def peek_next_time(self) -> Optional[float]:
        """Peek at the next time in the queue without removing it.
        Returns None if the queue is empty.
        """
        if not self.queue:
            return None
        return self.queue[0][0]

    def __len__(self) -> int:
        
        return len(self.queue)
    
    def pop_all(self) -> np.ndarray:
        """Pop all measurements from the queue.
        Returns the popped measurements as a NumPy array.
        """
        popped_measurements = list(self.queue)
        self.queue.clear()
        return np.array(popped_measurements)
    