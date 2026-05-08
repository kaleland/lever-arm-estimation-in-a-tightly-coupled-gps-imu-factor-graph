from abc import ABC, abstractmethod
from numpy import ndarray
import numpy as np
import scipy.sparse as sp
from factors import Factor, prior
from util import sparsify_dense_matrix, householder_qr, marginal_covariance_sparse
from util.constants import I3
from typing import Dict, Any, List, Tuple
from functools import partial
import copy
from dataclasses import dataclass
from itertools import count
from factors import marginalization,RobustFactor
from graphs import FactorGraph


class RobustFactorGraph(FactorGraph):

    def gen_y_and_scales(self)-> Tuple[np.ndarray,np.ndarray]:
        
        self.factors.sort(key=lambda f: f.time)

        n_rows = sum(factor.n_rows for factor in self.factors)
        y = np.zeros(n_rows, dtype=np.float64)
        scales = np.ones_like(y, dtype=np.float64)
        next_row_index = 0
        for factor in self.factors:
            # Get the state keys for the current factor
            state_keys = factor.states
            # Calculate the error
            error = factor.error_func(state_tuple = self.state_tuple_from_keys(state_keys))
            # Assign the error to the corresponding rows in y
            y[next_row_index:next_row_index + factor.n_rows] = error
            
            if isinstance(factor, RobustFactor):
                # Calculate the scales using the robust weight function
                scales[next_row_index:next_row_index + factor.n_rows] = factor.robust_weight_func(error)

            next_row_index += factor.n_rows


        self.y = y
        self.scales = scales

        return y, scales
    
    def gen_y(self) -> np.ndarray:
        
        self.gen_y_and_scales()
        return self.y.copy()

    # Marginalization and factor management
    def marginalize_states(self, keys: List[str], marginalization_time: float = 0.0):
        
        keys_to_remove = list(set(keys)) # Remove redundant keys
        keys_to_remove = self.order_state_keys(keys_to_remove)  # Sort keys by time and then by character part

        # Find all the factors that depend on the keys to be marginalized and
        # those factors rows in the L matrix
        factors_to_linearize = np.empty(len(self.factors), dtype = Factor)
        rows_to_linearize = np.empty(self.L.shape[0],dtype = int)
        kept_keys = []
        row_counter = 0
        next_rows_to_linearize_idx = 0
        next_factors_to_linearize_idx = 0
        for f in self.factors:
            if any(key in f.states for key in keys_to_remove):
                factors_to_linearize[next_factors_to_linearize_idx] = f
                rows_to_linearize[next_rows_to_linearize_idx:next_rows_to_linearize_idx + f.n_rows] = np.arange(row_counter, row_counter + f.n_rows)
                for key in f.states:
                    if key not in keys_to_remove and key not in kept_keys:
                        kept_keys.append(key)                
                next_factors_to_linearize_idx += 1
                next_rows_to_linearize_idx += f.n_rows

            row_counter += f.n_rows

        kept_keys = self.order_state_keys(kept_keys)  # Sort keys by time and then by character part

        # Trim the arrays to the actual size
        factors_to_linearize = factors_to_linearize[:next_factors_to_linearize_idx]
        rows_to_linearize = rows_to_linearize[:next_rows_to_linearize_idx]

        state_index_dict = self.gen_state_index_dict()

        # Find the columns of all the states that will be removed
        cols_to_remove = np.hstack([np.arange(state_index_dict[key],state_index_dict[key]+self.state_dict[key].delta_dimensions) for key in keys_to_remove if key in state_index_dict])
        # Find the columns of all the states that will be kept
        cols_to_keep = np.hstack([np.arange(state_index_dict[key],state_index_dict[key]+self.state_dict[key].delta_dimensions) for key in kept_keys if key in state_index_dict])


        # Create a sub-matrix of L that contains the intersection all the rows to linearize and
        # all the columns to remove and all the columns to keep (in that order)
        scaling_matrix = sp.diags(self.scales)
        L_scaled_sub = (scaling_matrix@self.L)[rows_to_linearize, :][:, np.concatenate((cols_to_remove, cols_to_keep))].toarray()
        y_scaled_sub = np.asarray((scaling_matrix@self.y)[rows_to_linearize]).flatten()

        R, QTy = householder_qr(L_scaled_sub, y_scaled_sub)
        n_dropped_columns = len(cols_to_remove)

        G = np.asarray(R[n_dropped_columns:, n_dropped_columns:])
        zm = np.asarray(QTy[n_dropped_columns:]).flatten()

        # Create x0, a tuple of the kept keys with their current state values
        x0 = tuple(copy.deepcopy(self.state_dict[key].value) for key in kept_keys)

        n_rows = G.shape[0]
        total_cols = np.sum([self.state_dict[key].delta_dimensions for key in kept_keys])

        self.add_factor(marginalization.MarginalizationFactor(
            time = marginalization_time,
            states = kept_keys,
            n_rows = n_rows,  # Number of rows in the new factor is the number of rows in G
            jacobians_size = n_rows * total_cols,  # Total number of elements in the Jacobian matrices
            error_func = partial(marginalization.marginalization_error, zm=zm, G=G, x0=x0),
            jacobian_func = partial(marginalization.marginalization_jacobians, G=G, x0=x0),
        ))

        # Now remove old factors and states
        self.remove_keys_and_factors(keys_to_remove)

    def marginalize_by_time(self, marginalization_time: float):
        """Marginalize states and factors based on measurement time using robust scaling.
        
        Inputs:
        - marginalization_time: Time threshold - factors before this time will be marginalized
        
        Returns:
        - None

        """
        # Identify factors with time before the marginalization time
        old_factors = [f for f in self.factors if f.time < marginalization_time]
        new_factors = [f for f in self.factors if f.time >= marginalization_time]
        
        if not old_factors:
            # No factors to marginalize
            return
        
        # Find all states connected to old factors
        old_factor_states = set()
        for factor in old_factors:
            old_factor_states.update(factor.states)
        
        # Find all states connected to new factors
        new_factor_states = set()
        for factor in new_factors:
            new_factor_states.update(factor.states)
        
        # States that can be dropped are those only connected to old factors
        states_to_drop = old_factor_states - new_factor_states
        
        # States that must be kept are those connected to both old and new factors
        states_to_keep = old_factor_states & new_factor_states
        
        if not states_to_drop:
            # No states can be marginalized
            return
        
        # Convert to sorted lists for consistent ordering
        keys_to_remove = self.order_state_keys(list(states_to_drop))
        kept_keys = self.order_state_keys(list(states_to_keep))
        
        # Generate the L matrix and y vector with scales if not already done
        if not hasattr(self, 'L') or not hasattr(self, 'y') or not hasattr(self, 'scales'):
            self.L = self.gen_L()
            self.y, self.scales = self.gen_y_and_scales()
        
        # Find rows corresponding to old factors that will be marginalized
        rows_to_linearize = []
        row_counter = 0
        for f in self.factors:
            if f.time < marginalization_time:
                rows_to_linearize.extend(range(row_counter, row_counter + f.n_rows))
            row_counter += f.n_rows
        
        if not rows_to_linearize:
            return
        
        rows_to_linearize = np.array(rows_to_linearize)
        
        state_index_dict = self.gen_state_index_dict()
        
        # Find the columns of all the states that will be removed
        cols_to_remove = np.hstack([
            np.arange(state_index_dict[key], state_index_dict[key] + self.state_dict[key].delta_dimensions) 
            for key in keys_to_remove if key in state_index_dict
        ])
        
        # Find the columns of all the states that will be kept
        cols_to_keep = np.hstack([
            np.arange(state_index_dict[key], state_index_dict[key] + self.state_dict[key].delta_dimensions) 
            for key in kept_keys if key in state_index_dict
        ]) if kept_keys else np.array([], dtype=int)
        
        if len(cols_to_keep) == 0:
            # All states from old factors can be dropped, no marginalization factor needed
            self.factors = new_factors
            self.remove_keys_and_factors(keys_to_remove)
            return
        
        # Create a sub-matrix of L with robust scaling applied
        scaling_matrix = sp.diags(self.scales)
        L_scaled_sub = (scaling_matrix @ self.L)[rows_to_linearize, :][:, np.concatenate((cols_to_remove, cols_to_keep))].todense()
        y_scaled_sub = (scaling_matrix @ self.y)[rows_to_linearize]
        
        # Perform QR decomposition for marginalization
        R, QTy = householder_qr(L_scaled_sub, y_scaled_sub)
        n_dropped_columns = len(cols_to_remove)
        
        G = R[n_dropped_columns:, n_dropped_columns:]
        zm = QTy[n_dropped_columns:]
        
        # Create x0, a tuple of the kept keys with their current state values
        x0 = tuple(copy.deepcopy(self.state_dict[key].value) for key in kept_keys)
        
        n_rows = G.shape[0]
        total_cols = np.sum([self.state_dict[key].delta_dimensions for key in kept_keys])
        
        # Remove all old factors
        self.factors = new_factors
        
        # Add marginalization factor if there are states to constrain
        if kept_keys:
            self.add_factor(Factor(
                time=marginalization_time,
                states=kept_keys,
                n_rows=n_rows,
                jacobians_size=n_rows * total_cols,
                error_func=partial(marginalization.marginalization_error, zm=zm, G=G, x0=x0),
                jacobian_func=partial(marginalization.marginalization_jacobians, G=G, x0=x0),
            ))
        
        # Remove the dropped states
        self.remove_keys_and_factors(keys_to_remove)

    def marginal_covariance_of_states(self, keys: list[str], safe=False) -> np.ndarray:
        """Calculate the marginal covariance of the specified states.

        Inputs:
        - keys: List of state keys for which to compute the marginal covariance

        Returns:
        - Covariance matrix as a numpy ndarray

        """
        if not safe:
            keys_for_covariance = list(set(keys)) # Keys that will be included in the covariance matrix
            keys_for_covariance = self.order_state_keys(keys_for_covariance)  # Sort keys by time and then by character part
            state_index_dict = self.gen_state_index_dict()
            cols_for_covariance = np.hstack([np.arange(state_index_dict[key],state_index_dict[key]+self.state_dict[key].delta_dimensions) for key in keys_for_covariance if key in state_index_dict])
            L_scaled = sp.diags(self.scales) @ self.L
            P = marginal_covariance_sparse(L_scaled, cols = cols_for_covariance)
            return P
        else:
            # Drop columns from self.L that have all zero entries
            non_zero_cols = np.where(self.L.getnnz(axis=0) > 0)[0]
            L_reduced = self.L[:, non_zero_cols]
            L_reduced_scaled = sp.diags(self.scales) @ L_reduced
            keys_for_covariance = list(set(keys)) # Keys that will be included in the covariance matrix
            keys_for_covariance = self.order_state_keys(keys_for_covariance)  # Sort keys
            state_index_dict = self.gen_state_index_dict()
            cols_for_covariance = np.hstack([np.arange(state_index_dict[key],state_index_dict[key]+self.state_dict[key].delta_dimensions) for key in keys_for_covariance if key in state_index_dict])
            # Map original columns to reduced columns
            col_mapping = {orig_col: new_col for new_col, orig_col in enumerate(non_zero_cols)}
            reduced_cols_for_covariance = [col_mapping[col] for col in cols_for_covariance if col in col_mapping]
            P_reduced = marginal_covariance_sparse(L_reduced_scaled, cols = reduced_cols_for_covariance)
            return P_reduced
       

    def schur_marginalization(self, keys, marginalization_time = 0):
        
        raise NotImplementedError("Schur marginalization is not implemented for RobustFactorGraph.")
