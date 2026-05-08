
import numpy as np
import scipy
from scipy.linalg import solve_triangular, qr
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la

def sparsify_dense_matrix(A: np.ndarray, start_row: int, start_col: int):
    """Produce the data, row, and column arrays to include the dense matrix A in a
    SciPy CSR sparse matrix.

    Inputs:
    A: np.ndarray, shape (n_rows, n_cols), dense matrix to sparsify
    start_row: int, starting row index in the sparse matrix
    start_col: int, starting column index in the sparse matrix

    Outputs:
    data: np.ndarray, shape (n_nonzero), data values of the sparse matrix
    row: np.ndarray, shape (n_nonzero), row indices of the data values
    col: np.ndarray, shape (n_nonzero), column indices of the data values
    """
    # If A is 1D (ie a vector), convert it to a 2D array with one row.
    if A.ndim == 1:
        A = A[np.newaxis,:]
    n_rows, n_cols = A.shape
    # data = A.flatten()
    # row = np.repeat(np.arange(n_rows),n_cols) + start_row
    # col = np.tile(np.arange(n_cols),n_rows) + start_col
    return A.flatten(), \
        np.repeat(np.arange(n_rows,dtype=int),n_cols) + start_row, \
        np.tile(np.arange(n_cols,dtype=int),n_rows) + start_col



def householder_qr(A, b):
    """Given A and b from Ax=b, this function returns R and y from the equivalent
    problem Rx=y, where R is upper triangular and y is the transformed b vector.
    """
    # Use numpy.linalg.qr for robustness and simplicity
    # mode='reduced' (default) returns Q with shape (M, K) and R with shape (K, N) where K = min(M, N)
    Q, R = np.linalg.qr(A)
    
    # y = Q^T * b
    # If b is 1D, Q.T @ b works correctly. If b is 2D (M, 1), it also works.
    y = Q.T @ b
    
    return R, y

# def marginal_covariance(L, cols)-> np.ndarray:
#     """
#     Compute the marginal covariance of a subset of state variables from
#     a linearized factor graph Jacobian.

#     Parameters
#     ----------
#     L : ndarray, shape (m, n)
#         Weighted Jacobian (W^(1/2) @ J)
#     cols : list or array-like
#         Indices of the state variables to compute marginal covariance for

#     Returns
#     -------
#     P_block : ndarray, shape (len(cols), len(cols))
#         Marginal covariance of the specified state variables
#     """
#     # QR decomposition of A
#     _, R = qr(L, mode='economic')  # R is upper-triangular

#     n = R.shape[0]  # total number of states
#     m = len(cols)   # number of states of interest
#     P_block = np.zeros((m, m))

#     # Compute marginal covariance using triangular solves
#     for i in range(m):
#         e = np.zeros(n)
#         e[cols[i]] = 1.0

#         # Solve R^T y = e
#         y = solve_triangular(R.T, e, lower=True)

#         # Solve R x = y
#         x = solve_triangular(R, y, lower=False)

#         # Extract the relevant block
#         P_block[i, :] = x[cols]

#     # Symmetrize for numerical stability
#     P_block = 0.5 * (P_block + P_block.T)
#     return P_block

def marginal_covariance(A, cols, tol_diag=1e-12, use_pinv_on_singular=True,report_diagnostics=False):
    """Compute marginal covariance of columns `cols` by reordering them to the end,
    QR'ing the design matrix, and using the bottom-right block of R.

    Parameters
    ----------
    A : (m, n) ndarray
        Weighted Jacobian (W^(1/2) @ J).
    cols : array-like of int
        Indices of the state variables to compute marginal covariance for.
    tol_diag : float
        Threshold for treating diagonal elements of R22 as zero (singular).
    use_pinv_on_singular : bool
        If True, fall back to pinv-based marginal covariance when R22 is singular.

    Returns
    -------
    P_block : (k, k) ndarray
        Marginal covariance for the variables in `cols` (in the same order).
    info : dict
        Diagnostics: {'R22_diag_min', 'is_singular', 'perm'}.

    """
    A = np.asarray(A)
    m, n = A.shape
    cols = list(cols)
    k = len(cols)
    if k == 0:
        return np.zeros((0,0)), {'R22_diag_min': None, 'is_singular': False, 'perm': None}

    # Build column permutation: keep columns not in cols first, then cols
    all_idx = list(range(n))
    other = [i for i in all_idx if i not in cols]
    perm = other + cols  # new column order
    
    # Check if permutation indices are valid
    if len(perm) > n or max(perm) >= n:
        print(f"Warning: Invalid permutation indices (max: {max(perm)}, matrix cols: {n})")
        print(f"Cols requested: {cols}")
        print(f"Matrix shape: {A.shape}")
        # Return zero covariance as fallback
        return np.zeros((k, k)) if not report_diagnostics else (np.zeros((k, k)), {'error': 'invalid_permutation'})
    
    inv_perm = np.argsort(perm)  # to map back if needed

    # Permute columns of A
    A_perm = A[:, perm]

    # QR decomposition: For marginal covariance, we need the bottom-right k x k block of R
    # This requires R to be at least (n-k+1) x n in shape, which means we need full QR when m < n-k+1
    if m >= n:
        # Economic mode works: R will be (n, n) 
        Q, R = qr(A_perm, mode='economic')
    else:
        # Use full QR to get sufficient rows in R matrix
        Q, R = qr(A_perm, mode='full')
        # R from full QR is (m, n), but we need at least (n, n) conceptually
        # The bottom rows of R beyond row m-1 would be zero, so we can pad if needed
        if m < n:
            # Pad R with zeros to make it (n, n)
            R_padded = np.zeros((n, n))
            R_padded[:m, :] = R
            R = R_padded

    # Extract R22: bottom-right k x k block
    R22 = R[n - k : n, n - k : n]

    # Check if R22 is empty or invalid
    if R22.size == 0 or k == 0:
        print(f"Warning: R22 matrix is empty (shape: {R22.shape}), returning zero covariance")
        print(f"  Debug: k={k}, n={n}, A_perm.shape={A_perm.shape}, R.shape={R.shape}")
        print(f"  Debug: cols={cols}, perm={perm}")
        print(f"  Debug: R matrix:\n{R}")
        if report_diagnostics:
            return np.zeros((k, k)), {'R22_diag_min': None, 'is_singular': True, 'perm': perm, 'empty_matrix': True}
        return np.zeros((k, k))

    # Diagnostics
    diag_min = np.min(np.abs(np.diag(R22)))
    # Also check condition number for better singularity detection
    cond_num = np.linalg.cond(R22)
    is_singular = diag_min <= tol_diag or cond_num > 1e12

    if is_singular and use_pinv_on_singular:
        # Fall back: compute (A^T A) pseudo-inverse and extract block
        Lambda = R.T @ R  # = A_perm^T A_perm
        # undo perm to get Lambda in original ordering
        # permuted-to-original: Lambda_orig = P * Lambda * P^T
        # Construct permutation matrix via indices:
        P_idx = np.array(perm)
        Lambda_orig = Lambda[np.ix_(np.argsort(P_idx), np.argsort(P_idx))]
        # compute pseudo-inverse and extract block
        Lambda_pinv = np.linalg.pinv(Lambda_orig)
        P_block = Lambda_pinv[np.ix_(cols, cols)]
        if report_diagnostics:
            return P_block, {'R22_diag_min': diag_min, 'is_singular': True, 'perm': perm}
        return P_block

    # If not singular: compute R22^{-1} via triangular solve and form P = X X^T
    # Solve R22 X = I  -> X = R22^{-1}
    I_k = np.eye(k)
    try:
        X = solve_triangular(R22, I_k, lower=False, check_finite=False)  # shape (k,k)
        P_block = X @ X.T  # equals R22^{-1} R22^{-T}
    except np.linalg.LinAlgError as e:
        # Handle case where solve_triangular fails due to singularity
        print(f"Warning: solve_triangular failed with {e}, falling back to pseudo-inverse")
        # Fall back: compute (A^T A) pseudo-inverse and extract block
        Lambda = R.T @ R  # = A_perm^T A_perm
        # undo perm to get Lambda in original ordering
        P_idx = np.array(perm)
        Lambda_orig = Lambda[np.ix_(np.argsort(P_idx), np.argsort(P_idx))]
        # compute pseudo-inverse and extract block
        Lambda_pinv = np.linalg.pinv(Lambda_orig)
        P_block = Lambda_pinv[np.ix_(cols, cols)]
        if report_diagnostics:
            return P_block, {'R22_diag_min': diag_min, 'is_singular': True, 'perm': perm, 'fallback_used': True}
        return P_block

    # P_block corresponds to the covariance of the last k variables in permuted ordering,
    # which are the original `cols` in the order provided.
    if report_diagnostics:
        return P_block, {'R22_diag_min': diag_min, 'is_singular': False, 'perm': perm}
    else:
        return P_block

import scipy.sparse as sp
import scipy.sparse.linalg as spla

def marginal_covariance_sparse(A: sp.csr_matrix, cols, regularize=0.0) -> np.ndarray:
    
    if not sp.isspmatrix_csr(A):
        A = A.tocsr()
    else:
        A = A

    m, n = A.shape

    cols = list(cols)
    k = len(cols)
    if k == 0:
        return np.zeros((0,0)), {'R22_diag_min': None, 'is_singular': False, 'perm': None}

    # Build column permutation: keep columns not in cols first, then cols
    all_idx = list(range(n))
    other = [i for i in all_idx if i not in cols]
    perm = other + cols  # new column order
    
    # Check if permutation indices are valid
    if len(perm) > n or max(perm) >= n:
        print(f"Warning: Invalid permutation indices (max: {max(perm)}, matrix cols: {n})")
        print(f"Cols requested: {cols}")
        print(f"Matrix shape: {A.shape}")
        # Return zero covariance as fallback
        return np.zeros((k, k))
    
    inv_perm = np.argsort(perm)  # to map back if needed

    # Permute columns of A
    A_perm = A[:, perm]
    
    if k <= 0 or k > n:
        raise ValueError("k must be 1..n")

    # Build sparse normal matrix Lambda = A^T A
    # Use (A.T @ A) which returns a CSR or CSC (symmetric)
    Lambda = (A_perm.T).dot(A_perm)  # typically returns a csr/csc matrix

    # Optionally regularize Lambda to improve stability
    if regularize and regularize > 0.0:
        Lambda = Lambda + regularize * sp.eye(n, format='csr')

    Lambda = Lambda.tocsc()  # Ensure CSC format for spla functions
    factor = spla.splu(Lambda)
    Lambda_inv_last = np.zeros((k,k))
    m = Lambda.shape[0]
    idx = np.arange(m - k, m) # Last k indices

    for j, col in enumerate(idx):
        e = np.zeros(m)
        e[col] = 1.0
        x = factor.solve(e)
        Lambda_inv_last[:, j] = x[idx]

    # Symmetrize for numerical stability
    P_block = 0.5 * (Lambda_inv_last + Lambda_inv_last.T)
    return P_block


def prefixed_counter(prefix: str, start: int = 0):
    """Create a counter that generates prefixed strings with an incrementing number.

    Args:
    - prefix (str): The prefix to use for the counter.
    - start (int): The starting number for the counter.

    Returns:
    - A generator that yields strings in the format "prefix_number".

    """
    count = start
    while True:
        yield f"{prefix}{count}"
        count += 1

def weight_from_covariance(cov: np.ndarray) -> np.ndarray:
    """Convert a covariance matrix to a weight matrix.
    
    Args:
    - cov (np.ndarray): The covariance matrix.
    
    Returns:
    - np.ndarray: The weight matrix, which is the inverse of the covariance matrix.

    """
    if np.isscalar(cov) or cov.ndim == 0:
        return 1.0 / np.sqrt(cov)
    return np.linalg.cholesky(np.linalg.inv(cov)).T#np.linalg.inv(np.linalg.cholesky(cov)).T

def efficient_weighted_least_squares(A: np.ndarray, b: np.ndarray, Sigma: sp.csr_matrix) -> np.ndarray:
    
    # From Google Gemini

    # --- Assume you have: ---
    # A: (N, m) dense np.ndarray with m << N
    # b: (N,) dense np.ndarray
    # Sigma: (N, N) sparse sp.csr_matrix

    # --- 1. Create the augmented right-hand side ---
    # We stack A and b horizontally to solve for them at the same time.
    # y.reshape(-1, 1) turns the (N,) vector into an (N, 1) column matrix.
    B = np.hstack((A, b.reshape(-1, 1)))

    # --- 2. Perform one sparse solve ---
    # This is the core of the solution.
    # S will be a dense (N, m+1) matrix where:
    # S = [Sigma_inv * A, Sigma_inv * b]
    S = spla.spsolve(Sigma, B)

    # --- 3. Separate the results ---
    # Z = Sigma_inv * A
    Z = S[:, :3]  
    # z = Sigma_inv * b
    z = S[:, 3]    

    # --- 4. Form the small (m, m) system ---
    # H = A.T @ Z  (which is A.T @ Sigma_inv @ A)
    H = A.T @ Z

    # g = A.T @ z  (which is A.T @ Sigma_inv @ b)
    g = A.T @ z

    # --- 5. Solve the dense (m, m) system ---
    # This is numerically more stable than using np.linalg.inv
    try:
        update_step = la.solve(H, -g, assume_a='sym')
    except la.LinAlgError:
        # Fallback if H is singular, e.g., using pseudo-inverse
        update_step = -la.pinv(H) @ g   

    return update_step 

def efficient_information_matrix(A: np.ndarray, Sigma: sp.csr_matrix) -> np.ndarray:
    
    # From Google Gemini
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    import scipy.linalg as la
    # --- Assume you have: ---
    # A: (N, m) dense np.ndarray with m << N
    # Sigma: (N, N) sparse sp.csr_matrix

    # --- 1. Perform one sparse solve ---
    # This is the core of the solution.
    # S will be a dense (N, m) matrix where:
    # S = Sigma_inv * A
    S = spla.spsolve(Sigma, A)

    # --- 2. Form the small (m, m) system ---
    # H = A.T @ S  (which is A.T @ Sigma_inv @ A)
    H = A.T @ S

    return H


def show_colored_matrix(matrix, cmap='viridis', value_fmt='{:.2f}', base_size=0.8):
    
    from matplotlib import pyplot as plt
    """
    Display a 2D matrix with color coding and numeric annotations.
    Figure size scales automatically with matrix dimensions.

    Parameters
    ----------
    matrix : np.ndarray
        2D array to display.
    cmap : str, optional
        Matplotlib colormap name (default 'viridis').
    value_fmt : str, optional
        Format string for values (default '{:.2f}').
    base_size : float, optional
        Controls how big each cell appears (default 0.8 inches per cell).
    """
    if matrix.ndim != 2:
        raise ValueError("Input must be a 2D matrix")

    nrows, ncols = matrix.shape
    fig_width = max(4, ncols * base_size)
    fig_height = max(3, nrows * base_size)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(matrix, cmap=cmap)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Value', rotation=-90, va="bottom")

    # Show all ticks and label them with their indices
    ax.set_xticks(np.arange(ncols))
    ax.set_yticks(np.arange(nrows))
    ax.set_xticklabels(np.arange(ncols))
    ax.set_yticklabels(np.arange(nrows))

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Adjust font size based on matrix size
    font_size = max(6, 16 - 0.3 * max(nrows, ncols))

    # Loop over data dimensions and create text annotations
    for i in range(nrows):
        for j in range(ncols):
            ax.text(j, i, value_fmt.format(matrix[i, j]),
                    ha="center", va="center", color="black", fontsize=font_size)

    ax.set_title("Matrix Visualization", fontsize=font_size + 2)
    fig.tight_layout()
    plt.show()