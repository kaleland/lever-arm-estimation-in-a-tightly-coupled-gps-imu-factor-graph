from graphs import FactorGraph, RobustFactorGraph
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import numpy as np
from matplotlib import pyplot as plt
import time

def GN_opt_damped(
        graph: FactorGraph, 
        print_iterations = True,
        time_iterations = False,
        max_iter: int = 10, 
        min_step_factor_power_2: int = 1,
        abs_decrease_tol: float = 1e-10,
        rel_decrease_tol: float = 1e-5,
        l2: bool = False) -> FactorGraph:
    
    robust_graph = isinstance(graph, RobustFactorGraph)
    robust_opt = robust_graph and not l2
    if robust_opt:
        old_y, scales = graph.gen_y_and_scales()

    else:
        old_y = graph.gen_y()

    L = graph.gen_L()
    # denseL = L.todense()
    # # put dense L in a csv
    # # np.savetxt('L.csv', denseL, delimiter=',')
    # # print(denseL)
    # plt.spy(L.todense())
    # print(np.linalg.matrix_rank(denseL))
    if time_iterations:
        start_time = time.time()



    for iter in range(max_iter):
        # delta = -1*np.linalg.lstsq(L.todense(), old_y, rcond=None)[0]
        # plt.figure()
        # plt.imshow(L.todense())
        # plt.colorbar()
        # # plt.figure()
        # # plt.imshow(old_y.reshape(-1,1))
        # # plt.colorbar()
        # plt.figure()
        # plt.spy(L.todense())
        # # plt.figure()
        # # plt.spy(old_y.reshape(-1,1))
        # plt.show()
        # # print('rank L', np.linalg.matrix_rank(L.todense()))
        
        # Identify all-zero columns in L
        # Use column norms to identify zero columns (more robust than sums)
        # For sparse matrices, this is more efficient than iterating through columns
        L_col_norms = np.sqrt(np.array((L.multiply(L)).sum(axis=0)).flatten())
        nonzero_cols = np.where(L_col_norms > 1e-15)[0]  # Indices of non-zero columns (with small tolerance)
        zero_cols = np.where(L_col_norms <= 1e-15)[0]    # Indices of all-zero columns
        
        if len(zero_cols) > 0:
            if print_iterations:
                print(f'Found {len(zero_cols)} all-zero columns in L matrix. Using truncated optimization.')
            # Create truncated L matrix with only non-zero columns
            L_trun = L[:, nonzero_cols]
            
            if robust_opt:
                scaling_matrix = sp.diags(scales)
                L_trun_scaled = scaling_matrix @ L_trun
                old_y_scaled = scaling_matrix @ old_y
                # Solve for truncated delta
                delta_trun = -1 * spla.spsolve(L_trun_scaled.T @ L_trun_scaled, L_trun_scaled.T @ old_y_scaled)
            else:
                # Solve for truncated delta
                delta_trun = -1 * spla.spsolve(L_trun.T @ L_trun, L_trun.T @ old_y)
            
            # Expand delta back to full size with zeros for removed columns
            delta = np.zeros(L.shape[1])
            delta[nonzero_cols] = delta_trun
        else:
            # No all-zero columns, proceed as before
            if robust_opt:
                scaling_matrix = sp.diags(scales)
                L_scaled = scaling_matrix @ L
                old_y_scaled = scaling_matrix @ old_y
                # Use the robust weight function to scale the error vector
                delta = -1 * spla.spsolve(L_scaled.T @ L_scaled, L_scaled.T @ old_y_scaled)
            else:
                delta = -1 * spla.spsolve(L.T @ L, L.T @ old_y)

        # Assert that delta doesn't contain any NaN values
        if np.any(np.isnan(delta)):
            pause_for_debugging_here = True
        assert not np.any(np.isnan(delta)), f"Delta contains NaN values: {delta}"

        graph.update_states(delta)
        
        if robust_opt:
            # Generate the new y vector using the updated states
            y, scales = graph.gen_y_and_scales()
            new_y_old_scales = scaling_matrix @ y
        else:
            y = graph.gen_y()



        for try_iter in range(1,min_step_factor_power_2+1):
            bad = (np.linalg.norm(y) > np.linalg.norm(old_y)) if not robust_opt else (np.linalg.norm(new_y_old_scales) > np.linalg.norm(old_y_scaled))
            if bad:
                graph.revert_states()

                delta = delta / (2**try_iter)
                # Assert that delta doesn't contain any NaN values
                assert not np.any(np.isnan(delta)), f"Delta contains NaN values after damping: {delta}"
                graph.update_states(delta)

                if robust_opt:
                    y, scales = graph.gen_y_and_scales()
                    new_y_old_scales = scaling_matrix @ y
                else:
                    y = graph.gen_y()
                # if try_iter == 5:
                #     print('Minimum step size reached. Exiting.')
            else:
                # print('Step size: 1/{:d}'.format(2**(try_iter-1)))
                break       

        if print_iterations:
            print('\nIteration {:d}'.format(iter))
            if robust_opt:
                print('Error before update (scaled): {:.3e} ({:.3e})'.format(np.linalg.norm(old_y), np.linalg.norm(old_y_scaled)))
                print('Error after update (scaled): {:.3e} ({:.3e})'.format(np.linalg.norm(y), np.linalg.norm(new_y_old_scales)))
            else:
                print('Error before update: {:.3e}'.format(np.linalg.norm(old_y)))
                print('Error after update: {:.3e}'.format(np.linalg.norm(y)))
            # # For debugging
            # print('y',y)
            # print('old_y',old_y)

        absolute_error_decrease = np.linalg.norm(old_y_scaled) - np.linalg.norm(new_y_old_scales) if robust_opt else np.linalg.norm(old_y) - np.linalg.norm(y)
        relative_error_decrease = absolute_error_decrease / np.linalg.norm(old_y_scaled)if robust_opt else absolute_error_decrease / np.linalg.norm(old_y)

        if print_iterations:
            print('Absolute error decrease: {:.3e}'.format(absolute_error_decrease))
            print('Relative error decrease: {:.3e}'.format(relative_error_decrease))
        
        if absolute_error_decrease < 0:
            if print_iterations:
                print('Converged: absolute error increase.')
                #print('State estimate with increased error: ',graph.state[0].p,graph.state[0].delta_tr)
                print('Reverting to previous state estimate.\n')
            graph.revert_states()
            graph.gen_y() if not robust_opt else graph.gen_y_and_scales()[0]
            graph.gen_L()
            break
        elif absolute_error_decrease < abs_decrease_tol:
            if print_iterations:
                print('Converged: absolute error decrease less than {:.3e}\n'.format(abs_decrease_tol))
            break
        elif relative_error_decrease < rel_decrease_tol:
            if print_iterations:
                print('Converged: relative error decrease less than {:.3e}\n'.format(rel_decrease_tol))
            break
        old_y = y.copy()

        L = graph.gen_L()
        # plt.spy(L.todense())

        if iter == max_iter-1:
            if print_iterations:
                print('Max iterations reached')
        if time_iterations:
            elapsed_time = time.time() - start_time
            print(f"Iteration {iter + 1}/{max_iter} took {elapsed_time:.4f} seconds")
            start_time = time.time()
    return graph

def gn_opt_damped_l2_then_robust(graph: RobustFactorGraph, **kwargs):
    
    if not isinstance(graph, RobustFactorGraph):
        GN_opt_damped(graph, l2=True, **kwargs)
        return graph
    else:
        GN_opt_damped(graph, l2=True, **kwargs)
        GN_opt_damped(graph, l2=False, **kwargs)
        return graph