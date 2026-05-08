from numpy import ndarray
import numpy as np
import scipy.sparse as sp
from factors import Factor, prior, between
from util import sparsify_dense_matrix, householder_qr, marginal_covariance_sparse
from util.constants import I3
from states import State
from typing import Dict, Any, List, Tuple
from functools import partial
import copy
from dataclasses import dataclass
from itertools import count
from factors import marginalization


class FactorGraph:

    def __init__(self):
        """Initialize."""
        self.state_dict: Dict[str,State] = {}
        self.factors: List[Factor] = []

    # Optimization
    def gen_L(self)-> sp.csr_matrix:
        
        # Sort the factors by time
        self.factors.sort(key=lambda f: f.time)

        # Initialize the arrays for the sparse matrix
        n_rows, n_cols, n_entries = self.calc_shape_L()
        data = np.zeros(n_entries, dtype=np.float64)
        row = np.zeros(n_entries, dtype=np.int32)
        col = np.zeros(n_entries, dtype=np.int32)
        next_data_index = 0
        next_row_index = 0
        state_index_dict = self.gen_state_index_dict()

        for factor in self.factors:
            total_jacobian_entries = 0
            # Get the state keys for the current factor
            state_keys = factor.states
            # Get the corresponding state indices
            state_start_col_idxs = [state_index_dict[key] for key in state_keys]
            # Calculate the error and Jacobians
            jacobians = factor.jacobian_func(state_tuple = self.state_tuple_from_keys(state_keys))

            for state_key_idx in range(len(state_keys)):
                jacobian_state = jacobians[state_key_idx] if len(state_keys)>1 else jacobians 
                state_dim = self.state_dict[state_keys[state_key_idx]].delta_dimensions
                
                if state_dim == 1:
                    if np.isscalar(jacobian_state):
                        # If the state and Jacobian are scalar, directly assign the value
                        data[next_data_index] = jacobian_state
                        row[next_data_index] = next_row_index
                        col[next_data_index] = state_start_col_idxs[state_key_idx]
                        next_data_index += 1
                        total_jacobian_entries += 1
                    else:
                        # If the state is a scalar but the Jacobian is an array, handle it as a column vector
                        n_jacobian_rows = jacobian_state.shape[0]
                        data[next_data_index:next_data_index + n_jacobian_rows] = jacobian_state.flatten()
                        row[next_data_index:next_data_index + n_jacobian_rows] = next_row_index + np.arange(n_jacobian_rows)
                        col[next_data_index:next_data_index + n_jacobian_rows] = state_start_col_idxs[state_key_idx]
                        next_data_index += n_jacobian_rows
                        total_jacobian_entries += n_jacobian_rows
                else:
                    subData, subRow, subCol = sparsify_dense_matrix(
                        jacobian_state, 
                        next_row_index, 
                        state_start_col_idxs[state_key_idx]
                    )
                    # Debug the problematic assignment
                    expected_size = len(subData)
                    available_size = len(data[next_data_index:next_data_index + expected_size])
                    if expected_size != available_size:
                        factor_type = type(factor).__name__
                        print(f"ERROR: Size mismatch in factor {factor_type}")
                        print(f"  jacobian_state shape: {jacobian_state.shape}")
                        print(f"  subData shape: {subData.shape}, len: {len(subData)}")
                        print(f"  expected slice size: {expected_size}, available: {available_size}")
                        print(f"  next_data_index: {next_data_index}, data array size: {len(data)}")
                    
                    data[next_data_index:next_data_index + len(subData)] = subData
                    row[next_data_index:next_data_index + len(subData)] = subRow
                    col[next_data_index:next_data_index + len(subData)] = subCol
                    next_data_index += len(subData)
                    total_jacobian_entries += len(subData)
            
            if total_jacobian_entries != factor.jacobians_size:
                factor_type = type(factor).__name__
                print(f"WARNING: Total Jacobian entries {total_jacobian_entries} does not match expected {factor.jacobians_size} in factor {factor_type}")    
            next_row_index += factor.n_rows

        # Create the sparse matrix
        nonzero_idxs = data != 0
        data = data[nonzero_idxs]
        row = row[nonzero_idxs]
        col = col[nonzero_idxs]

        self.L = sp.csr_matrix((data, (row, col)), shape=(n_rows, n_cols))

        return self.L

    def gen_y(self)-> np.ndarray:
        
        self.factors.sort(key=lambda f: f.time)

        n_rows = sum(factor.n_rows for factor in self.factors)
        y = np.zeros(n_rows, dtype=np.float64)
        next_row_index = 0
        for factor in self.factors:
            # Get the state keys for the current factor
            state_keys = factor.states
            # Calculate the error
            error = factor.error_func(state_tuple = self.state_tuple_from_keys(state_keys))
            # Assign the error to the corresponding rows in y
            y[next_row_index:next_row_index + factor.n_rows] = error
            next_row_index += factor.n_rows

        self.y = y

        return y
    
    def update_states(self, delta: np.ndarray):
        """Update the states in the graph with the given delta vector.
        The delta vector is expected to be in the same order as the state keys 
        in the state_index_dict.
        """
        # # Store a copy of the current state_dict before updating
        # self.old_state_dict = copy.deepcopy(self.state_dict)

        # Apply the delta to the state_dict
        state_index_dict = self.gen_state_index_dict()
        for key, index in state_index_dict.items():
            state_dim = self.state_dict[key].delta_dimensions

            # Update the state with the corresponding delta
            if state_dim == 1:
                self.state_dict[key].update(delta[index])
            else:
                self.state_dict[key].update(delta[index:index + state_dim])

    def revert_states(self):
        """Revert the states in the graph to the previous state_dict.
        This is useful if the optimization step needs to be rolled back.
        """
        # self.state_dict = copy.deepcopy(self.old_state_dict)   
        for key in self.state_dict:
            self.state_dict[key].revert()
        
    # Marginalization and factor management
    def remove_factors_before_time(self, time: float):
        """Remove all factors before a certain time."""
        self.factors = [f for f in self.factors if f[0] >= time]

    def remove_keys_and_factors(self, keys: List[str]):
        """Remove all factors that contain any of the specified keys."""
        self.factors = [f for f in self.factors if not any(key in f.states for key in keys)]
        # Also remove the states from the state_dict
        for key in keys:
            if key in self.state_dict:
                del self.state_dict[key]

    def marginalize_states(self, keys: List[str], marginalization_time: float = 0.0):
        
        keys_to_remove = list(set(keys)) # Remove redundant keys
        keys_to_remove = self.order_state_keys(keys_to_remove)



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
        L_sub = self.L[rows_to_linearize, :][:, np.concatenate((cols_to_remove, cols_to_keep))].todense()
        y_sub = self.y[rows_to_linearize]

        R, QTy = householder_qr(L_sub, y_sub)
        n_dropped_columns = len(cols_to_remove)

        G = R[n_dropped_columns:, n_dropped_columns:]
        zm = QTy[n_dropped_columns:]
 

        # Create x0, a tuple of the kept keys with their current state values
        x0 = tuple(copy.deepcopy(self.state_dict[key].value) for key in kept_keys)

        n_rows = G.shape[0]
        total_cols = np.sum([self.state_dict[key].delta_dimensions for key in kept_keys])

        # Now remove all of the factors that contain the keys to be marginalized
        self.factors = [f for f in self.factors if not any(key in f.states for key in keys_to_remove)]
            

        self.add_factor(marginalization.MarginalizationFactor(
            time = marginalization_time,
            states = kept_keys,
            n_rows = n_rows,  # Number of rows in the new factor is the number of rows in G
            jacobians_size = n_rows * total_cols,  # Total number of elements in the Jacobian matrices
            error_func = partial(marginalization.marginalization_error, zm=zm, G=G, x0=x0),
            jacobian_func = partial(marginalization.marginalization_jacobians, G=G, x0 = x0),
        ))

        # Now remove old factors and states
        self.remove_keys_and_factors(keys_to_remove)

    def schur_marginalization(self, keys: List[str], marginalization_time: float = 0.0):
        
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

        L_sub = self.L[rows_to_linearize, :]
        y_sub = self.y[rows_to_linearize]
        Aa = np.asarray(L_sub[:, cols_to_remove].todense())
        Ab = np.asarray(L_sub[:, cols_to_keep].todense())
        b = y_sub.copy()
        H_aa = Aa.T @ Aa
        H_ab = Aa.T @ Ab
        H_bb = Ab.T @ Ab
        g_a  = Aa.T @ b
        g_b  = Ab.T @ b      
        H_marg = H_bb - H_ab.T @ np.linalg.inv(H_aa) @ H_ab
        g_marg = g_b - H_ab.T @ np.linalg.inv(H_aa) @ g_a

        L_chol = np.linalg.cholesky(H_marg)  # Lower triangular
        A_marg_equiv = L_chol.T              # We use the *upper* triangular Cholesky factor
        b_marg_equiv = np.linalg.solve(L_chol, g_marg)       

        G = A_marg_equiv.copy()
        zm = b_marg_equiv.copy()
        # G = H_marg.copy()
        # zm = g_marg.copy()

        # Create x0, a tuple of the kept keys with their current state values
        x0 = tuple(copy.deepcopy(self.state_dict[key].value) for key in kept_keys)
        # x0 = tuple(copy.deepcopy(self.state_dict[key]) for key in keys_to_remove+kept_keys)

        n_rows = G.shape[0]
        total_cols = np.sum([self.state_dict[key].delta_dimensions for key in kept_keys])
        # total_cols = np.sum([self.state_dim_dict[key[0]] for key in keys_to_remove+kept_keys])
            

        self.add_factor(Factor(
            time = marginalization_time,
            states = kept_keys,
            n_rows = n_rows,  # Number of rows in the new factor is the number of rows in G
            jacobians_size = n_rows * total_cols,  # Total number of elements in the Jacobian matrices
            error_func = partial(marginalization.marginalization_error, zm=zm, G=G, x0=x0),
            jacobian_func = partial(marginalization.marginalization_jacobians, G=G, x0 = x0),
        ))

        # Now remove old factors and states
        self.remove_keys_and_factors(keys_to_remove)

    def marginalize_by_time(self, marginalization_time: float):
        """Marginalize states and factors based on measurement time.
        
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
        # (i.e., they appear in old factors being marginalized AND in new factors)
        states_to_keep = old_factor_states & new_factor_states
        
        if not states_to_drop:
            # No states can be marginalized
            return
        
        # Convert to sorted lists for consistent ordering
        keys_to_remove = self.order_state_keys(list(states_to_drop))
        kept_keys = self.order_state_keys(list(states_to_keep))
        
        # Generate the L matrix and y vector if not already done
        if not hasattr(self, 'L') or not hasattr(self, 'y'):
            self.L = self.gen_L()
            self.y = self.gen_y()
        
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
        
        # Create a sub-matrix of L that contains the intersection of all rows to linearize
        # and all columns to remove and keep (in that order)
        L_sub = self.L[rows_to_linearize, :][:, np.concatenate((cols_to_remove, cols_to_keep))].todense()
        y_sub = self.y[rows_to_linearize]
        
        # Perform QR decomposition for marginalization
        R, QTy = householder_qr(L_sub, y_sub)
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

    def marginal_covariance_of_states(self, keys: list[str],safe=False) -> np.ndarray:
        
        if not safe:
            keys_for_covariance = list(set(keys)) # Keys that will be included in the covariance matrix
            keys_for_covariance = self.order_state_keys(keys_for_covariance)  # Sort keys by time and then by character part
            state_index_dict = self.gen_state_index_dict()
            cols_for_covariance = np.hstack([np.arange(state_index_dict[key],state_index_dict[key]+self.state_dict[key].delta_dimensions) for key in keys_for_covariance if key in state_index_dict])
            P = marginal_covariance_sparse(self.L, cols = cols_for_covariance)
            return P
        
        else:
            # Drop columns from self.L that have all zero entries
            non_zero_cols = np.where(self.L.getnnz(axis=0) > 0)[0]
            L_reduced = self.L[:, non_zero_cols]
            keys_for_covariance = list(set(keys)) # Keys that will be included in the covariance matrix
            keys_for_covariance = self.order_state_keys(keys_for_covariance)  # Sort keys
            state_index_dict = self.gen_state_index_dict()
            cols_for_covariance = np.hstack([np.arange(state_index_dict[key],state_index_dict[key]+self.state_dict[key].delta_dimensions) for key in keys_for_covariance if key in state_index_dict])
            # Map original columns to reduced columns
            col_mapping = {orig_col: new_col for new_col, orig_col in enumerate(non_zero_cols)}
            reduced_cols_for_covariance = [col_mapping[col] for col in cols_for_covariance if col in col_mapping]
            P_reduced = marginal_covariance_sparse(L_reduced, cols = reduced_cols_for_covariance)
            return P_reduced


    
    # Adding factors to the graph
    def add_factor(self, factor: Factor):
        
        self.factors.append(factor)

    # State management
    def add_state(self, key, state: State):
        
        if key in self.state_dict:
            raise KeyError("Key exists already")
        self.state_dict[key] = state
        
    def update_state(self, key, value):
        """Update the state with the given key to the new value.
        If the key does not exist, it will be added.
        """
        if key in self.state_dict:
            self.state_dict[key].value = value
        raise KeyError(f"State key '{key}' does not exist in the graph. Use add_state() to add it first.")

    def drop_unconstrained_states(self):
        """Drop all states that are not constrained by any factors.
        This is useful for cleaning up the graph after optimization.
        """
        constrained_keys = set()
        for factor in self.factors:
            constrained_keys.update(factor.states)

        # Remove all states that are not in the constrained keys
        self.state_dict = {key: value for key, value in self.state_dict.items() if key in constrained_keys}

    # Basic factors

    def add_prior_scalar_factor(self, time, prior_value, state_key, weight=1.0):
        """Add a scalar prior factor to the graph."""
        self.add_factor(Factor(
            time=time,  # Prior factors typically have no specific time
            states=[state_key],
            n_rows=1,  # One row for the scalar prior error
            jacobians_size=1,  # One element in the Jacobian for the scalar state
            error_func=partial(prior.prior_scalar_error, prior = prior_value, weight=weight),
            jacobian_func=partial(prior.prior_scalar_jacobian, weight=weight)
        ))

    def add_prior_r3_factor(self, time, prior_value, state_key, weight=I3):
        """Add a 3D vector prior factor to the graph."""
        self.add_factor(Factor(
            time=time,  # Prior factors typically have no specific time
            states=[state_key],
            n_rows=3,  # Three rows for the 3D vector prior error
            jacobians_size=9,  # 9 elements in the Jacobian for the 3D vector state
            error_func=partial(prior.prior_r3_error, prior = prior_value, weight=weight),
            jacobian_func=partial(prior.prior_r3_jacobian, weight=weight)
        ))

    def add_prior_so3_factor(self, time, prior_value, state_key, weight=I3):
        """Add a SO(3) prior factor to the graph."""
        self.add_factor(Factor(
            time=time,  # Prior factors typically have no specific time
            states=[state_key],
            n_rows=3,  # Three rows for the SO(3) prior error
            jacobians_size=9,  # 9 elements in the Jacobian for the SO(3) state
            error_func=partial(prior.prior_so3_error, prior = prior_value, weight=weight),
            jacobian_func=partial(prior.prior_so3_jacobian, prior=prior_value, weight=weight)
        ))

    def add_between_r3_factor(self, time, state_key1, state_key2, weight=I3):
        """Add a 3D vector between factor to the graph."""
        self.add_factor(between.BetweenR3Factor(
            time=time,
            states=[state_key1, state_key2],
            n_rows=3,  # Three rows for the 3D vector between error
            jacobians_size=18,  # 9 elements in the Jacobian for each of the two 3D vector states
            error_func=partial(between.between_r3_error, weight=weight),
            jacobian_func=partial(between.between_r3_jacobian, weight=weight)
        ))

    def add_between_so3_factor(self, time, state_key1, state_key2, weight=I3):
        """Add a SO(3) between factor to the graph."""
        self.add_factor(Factor(
            time=time,
            states=[state_key1, state_key2],
            n_rows=3,  # Three rows for the SO(3) between error
            jacobians_size=18,  # 9 elements in the Jacobian for each of the two SO(3) states
            error_func=partial(between.between_so3_error, weight=weight),
            jacobian_func=partial(between.between_so3_jacobian, weight=weight)
        ))

    # Utility functions for the graph
    def state_tuple_from_keys(self, key_list):
        
        return tuple(self.state_dict[key].value for key in key_list)

    def calc_shape_L(self):
        """Calculate the shape of the L matrix based on the factors in the graph.
        The number of rows is the sum of n_rows for all factors, and the number of columns
        is the total number of state dimensions.
        """
        n_rows = sum(factor.n_rows for factor in self.factors)
        n_cols = sum(self.state_dict[key].delta_dimensions for key in self.state_dict.keys())
        n_entries = sum(factor.jacobians_size for factor in self.factors)
        return (n_rows, n_cols, n_entries)

    def order_state_keys(self, state_key_list) -> List[str]:
        """Order state keys by their state.time first, then alphabetically by key.
        
        Inputs:
        - state_key_list: List or iterable of state keys to order
        
        Returns:
        - List of ordered state keys

        """
        return sorted(state_key_list, key=lambda x: (self.state_dict[x].time, x))

    def gen_state_index_dict(self):
        """Generate a dictionary mapping state keys to their corresponding indices in the state vector.
        The states are sorted by their state.time, and then alphabetically by their key.
        """
        state_keys = self.order_state_keys(self.state_dict.keys())
        state_index_dict = {}
        index = 0
        for key in state_keys:
            state_index_dict[key] = index
            index += self.state_dict[key].delta_dimensions
        return state_index_dict
 