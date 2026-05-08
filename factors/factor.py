from typing import Callable, Any, List
from dataclasses import dataclass

@dataclass
class Factor:
    time: float
    states: List[str]
    n_rows: int
    jacobians_size: int  # Total number of elements in the Jacobian matrices
    error_func: Callable[..., Any]
    jacobian_func: Callable[..., Any]

# from abc import ABC, abstractmethod
# import numpy as np

# class Factor(ABC):
#     '''
#     Abstract base class for factors
#     '''
#     _meas: float | np.ndarray
#     _states = []
#     _error: float | np.ndarray
#     _jacobians: list | np.ndarray
#     need_to_update_error: bool = True
#     need_to_update_jacobians: bool = True



#     @abstractmethod
#     def __init__(self, *args, **kwargs):
#         pass

#     @staticmethod
#     @abstractmethod
#     def _calc_error(meas, states: list, weight: np.ndarray|float, **kwargs):
#         '''
#         Compute the error of the factor
#         '''
#         pass

#     @staticmethod
#     @abstractmethod
#     def _calc_jacobians(meas, states: list, weight: np.ndarray|float, **kwargs):
#         '''
#         Compute the Jacobians of the factor
#         '''
#         pass

#     @property
#     def meas(self):
#         '''
#         Get the measurement of the factor
#         '''
#         return self._meas

#     @meas.setter
#     def meas(self, meas: float | np.ndarray):
#         '''
#         Set the measurement of the factor
#         '''
#         self._meas = meas
#         self.need_to_update_error = True
#         self.need_to_update_jacobians = True

#     @property
#     def states(self):
#         '''
#         Get the states of the factor
#         '''
#         return self._states
    
#     @states.setter
#     def states(self, states: list):
#         '''
#         Set the states of the factor
#         '''
#         self._states = states
#         self.need_to_update_error = True
#         self.need_to_update_jacobians = True


#     @property
#     def error(self):
#         '''
#         Get the error of the factor
#         '''
#         # check if force_update is set to True in kwargs
#         if self.need_to_update_error is False:
#             return self._error
#         self._error = self._calc_error(self._meas, self._states)  
#         self.need_to_update_error = False
        
#         return self._error
    
#     @property
#     def jacobians(self):
#         '''
#         Get the Jacobian of the factor
#         '''
#         # check if force_update is set to True in kwargs
#         if self.need_to_update_jacobians is False:
#             return self._jacobians
#         self._jacobians = self._calc_jacobians(self._meas, self._states)
#         self.need_to_update_jacobians = False
        
#         return self._jacobians
    