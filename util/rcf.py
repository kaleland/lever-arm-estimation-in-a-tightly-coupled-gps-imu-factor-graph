
import numpy as np
from typing import Union, Callable
from functools import partial


def huber_loss_function(error: Union[float, list, np.ndarray], k: float = 1.0) -> Union[float, np.ndarray]:
    """Huber loss function for robust error handling.
    If |x| <= k, rho(x) = x^2/2. If |x| > k, rho(x) = k*(|x| - k/2).

    Inputs:
    - error: A float or a list/array of floats representing error(s).
    - k: The threshold for the Huber function, default is 1.0.

    Returns:
    - A float or NumPy array of Huber loss values.

    """
    error = np.asarray(error)  # converts float, list, or array into a NumPy array

    abs_error = np.abs(error)
    loss = np.where(
        abs_error <= k,
        0.5 * error**2,  # Quadratic for small errors
        k * (abs_error - 0.5 * k)  # Linear for large errors
    )

    # Return scalar if input was scalar
    if np.isscalar(error) or np.ndim(error) == 0:
        return float(loss)
    else:
        return loss
    
def huber_influence(error: Union[float, list, np.ndarray], k: float = 1.0) -> Union[float, np.ndarray]:
    """Huber influence function for optimization (derivative of the Huber loss).

    Influence function: phi(e) = e for |e| ≤ k, phi(e) = k*sign(e) for |e| > k

    Inputs:
    - error: A float or a list/array of floats representing error(s).
    - k: The threshold for the Huber function, default is 1.0.

    Returns:
    - A float or NumPy array of influence values.

    """
    error = np.asarray(error)  # converts float, list, or array into a NumPy array

    abs_error = np.abs(error)
    influence = np.where(
        abs_error <= k,
        error,  # Linear for small errors
        k * np.sign(error)  # Constant for large errors
    )

    # Return scalar if input was scalar
    if np.isscalar(error) or np.ndim(error) == 0:
        return float(influence)
    else:
        return influence
    
def huber_sqrt_weight(error: Union[float, list, np.ndarray], k: float = 1.0) -> Union[float, np.ndarray]:
    """Huber sqrt-weight function for robust error handling.

    This implementation gives proper Huber cost optimization.

    It is sqrt(influence(x)/x).
    In the context of factor graph optimization, this means weighting both the
    residual and the design matrix by w(e).

    Huber influence function: phi(e) = e for |e| ≤ k, phi(e) = k*sign(e) for |e| > k
    Huber sqrt-weight function: w(e) = sqrt(phi(e)/e)
        = 1 for |e| ≤ k, sqrt(k/|e|) for |e| > k

    Inputs:
    - error: A float or a list/array of floats representing error(s).
    - k: The threshold for the Huber function, default is 1.0.

    Returns:
    - A float or NumPy array of weights that give proper Huber cost with (w*e)^2.

    """
    error = np.asarray(error)  # converts float, list, or array into a NumPy array

    abs_error = np.abs(error)
    
    # For |e| <= k: cost should be e^2, so w = 1
    # For |e| > k: cost should be k|e|, so w = sqrt(k/|e|)
    weights = np.where(
        abs_error <= k, 
        1.0,  # Gives cost = e^2 for small errors
        np.sqrt(k / (abs_error + 1e-10))  # Gives cost = k|e| for large errors
    )

    # Return scalar if input was scalar
    if np.isscalar(error) or np.ndim(error) == 0:
        return float(weights)
    else:
        return weights
    
def dcs_loss_function(error: Union[float, list, np.ndarray], k: float = 1.0) -> Union[float, np.ndarray]:
    """Dynamic Covariance Scaling (DCS) loss function for robust error handling.
    rho(x) = (s*error)^2/2 where s = min(1, 2*k/(error^2 + k).

    Inputs:
    - error: A float or a list/array of floats representing error(s).
    - k: The "Phi" for the DCS function, default is 1.0.

    Returns:
    - A float or NumPy array of DCS loss values.

    """
    error = np.asarray(error)  # converts float, list, or array into a NumPy array

    abs_error = np.abs(error)
    # scale = np.minimum(1.0, (2 * k) / (abs_error**2 + k))
    scale = np.minimum(1.0, np.sqrt((2 * k) / (abs_error**2 + k)))
    loss = (scale * error)**2/2

    # Return scalar if input was scalar
    if np.isscalar(error) or np.ndim(error) == 0:
        return float(loss)
    else:
        return loss
    
def dcs_influence(error: Union[float, list, np.ndarray], k: float = 1.0) -> Union[float, np.ndarray]:
    """Dynamic Covariance Scaling (DCS) influence function for optimization.
    phi(e) = d(rho)/d(e) = s^2 * e where s = min(1, 2*k/(e^2 + k).

    Inputs:
    - error: A float or a list/array of floats representing error(s).
    - k: The "Phi" for the DCS function, default is 1.0.

    Returns:
    - A float or NumPy array of DCS influence values.

    """
    error = np.asarray(error)  # converts float, list, or array into a NumPy array

    abs_error = np.abs(error)
    scale = np.minimum(1.0, (2 * k) / (abs_error**2 + k))
    influence = (scale**2) * error

    # Return scalar if input was scalar
    if np.isscalar(error) or np.ndim(error) == 0:
        return float(influence)
    else:
        return influence
    
    
def dcs_sqrt_weight(error: Union[float, list, np.ndarray], k: float) -> Union[float, np.ndarray]:
    """Dynamic Covariance Scaling sqrt-weight function.
    
    This implementation gives proper DCS loss optimization.
    It is sqrt(influence(x)/x).
    In the context of factor graph optimization, this means weighting both the
    residual and the design matrix by w(e).

    DCS sqrt-weight function: w(e) = np.minimum(1.0, (2 * k) / (e**2 + k)) (from Eq. 15 in  Robust Map Optimization Using Dynamic Covariance Scaling)

    Inputs:
    - error: A float or a list/array of floats representing error(s).
    - k: The "Phi" for the DCS function.

    Returns:
    - A float or NumPy array of weights that give proper DCS cost with (w*e)^2.

    """
    error = np.asarray(error)  # converts float, list, or array into a NumPy array

    weights = np.minimum(1.0, (2 * k) / (error**2 + k))

    # Return scalar if input was scalar
    if np.isscalar(error) or np.ndim(error) == 0:
        return float(weights)
    else:
        return weights
    
def cauchy_loss_function(error: Union[float, list, np.ndarray], k: float = 1.0) -> Union[float, np.ndarray]:
    """Cauchy loss function for robust error handling.
    rho(x) = (k^2/2) * log(1 + (x/k)^2).

    Inputs:
    - error: A float or a list/array of floats representing error(s).
    - k: The scale parameter for the Cauchy function, default is 1.0.

    Returns:
    - A float or NumPy array of Cauchy loss values.

    """
    error = np.asarray(error)  # converts float, list, or array into a NumPy array

    loss = (k**2 / 2) * np.log1p((error / k)**2)

    # Return scalar if input was scalar
    if np.isscalar(error) or np.ndim(error) == 0:
        return float(loss)
    else:
        return loss
    
def cauchy_influence(error: Union[float, list, np.ndarray], k: float = 1.0) -> Union[float, np.ndarray]:
    """Cauchy influence function for optimization (derivative of the Cauchy loss).

    Influence function: phi(e) = d(rho)/d(e) = e / (1 + (e/k)^2)

    Inputs:
    - error: A float or a list/array of floats representing error(s).
    - k: The scale parameter for the Cauchy function, default is 1.0.

    Returns:
    - A float or NumPy array of influence values.

    """
    error = np.asarray(error)  # converts float, list, or array into a NumPy array

    influence = error / (1 + (error / k)**2)

    # Return scalar if input was scalar
    if np.isscalar(error) or np.ndim(error) == 0:
        return float(influence)
    else:
        return influence
    
def cauchy_sqrt_weight(error: Union[float, list, np.ndarray], k: float = 1.0) -> Union[float, np.ndarray]:
    """Cauchy sqrt-weight function for robust error handling.

    This implementation gives proper Cauchy loss optimization.
    It is sqrt(influence(x)/x).
    In the context of factor graph optimization, this means weighting both the
    residual and the design matrix by w(e).

    Cauchy influence function: phi(e) = e / (1 + (e/k)^2)
    Cauchy sqrt-weight function: w(e) = sqrt(phi(e)/e) = 1 / sqrt(1 + (e/k)^2)

    Inputs:
    - error: A float or a list/array of floats representing error(s).
    - k: The scale parameter, 'c' for the Cauchy function, default is 1.0.

    Returns:
    - A float or NumPy array of weights that give proper Cauchy cost with (w*e)^2.

    """
    error = np.asarray(error)  # converts float, list, or array into a NumPy array

    weights = 1 / np.sqrt(1 + (error / k)**2)

    # Return scalar if input was scalar
    if np.isscalar(error) or np.ndim(error) == 0:
        return float(weights)
    else:
        return weights

def tukey_loss_function(error: Union[float, list, np.ndarray], k: float = 1.0) -> Union[float, np.ndarray]:
    """Tukey loss function for robust error handling.
    rho(x) = (k^2/6) * (1 - (1 - (x/k)^2)^3) for |x| <= k, else rho(x) = k^2/6.

    Inputs:
    - error: A float or a list/array of floats representing error(s).
    - k: The scale parameter for the Tukey function, default is 1.0.

    Returns:
    - A float or NumPy array of Tukey loss values.

    """
    error = np.asarray(error)  # converts float, list, or array into a NumPy array

    abs_error = np.abs(error)
    loss = np.where(
        abs_error <= k,
        (k**2 / 6) * (1 - (1 - (error / k)**2)**3),  # Tukey loss for small errors
        k**2 / 6  # Constant for large errors
    )

    # Return scalar if input was scalar
    if np.isscalar(error) or np.ndim(error) == 0:
        return float(loss)
    else:
        return loss
    
def tukey_influence(error: Union[float, list, np.ndarray], k: float = 1.0) -> Union[float, np.ndarray]:
    """Tukey influence function for optimization (derivative of the Tukey loss).

    Influence function: phi(e) = e * (1 - (e/k)^2)^2 for |e| ≤ k, else phi(e) = 0

    Inputs:
    - error: A float or a list/array of floats representing error(s).
    - k: The scale parameter for the Tukey function, default is 1.0.

    Returns:
    - A float or NumPy array of influence values.

    """
    error = np.asarray(error)  # converts float, list, or array into a NumPy array

    abs_error = np.abs(error)
    influence = np.where(
        abs_error <= k,
        error * (1 - (error / k)**2)**2,  # Tukey influence for small errors
        0.0  # Zero for large errors
    )

    # Return scalar if input was scalar
    if np.isscalar(error) or np.ndim(error) == 0:
        return float(influence)
    else:
        return influence

def tukey_sqrt_weight(error: Union[float, list, np.ndarray], k: float = 1.0) -> Union[float, np.ndarray]:
    """Tukey sqrt-weight function for robust error handling.
    
    This implementation gives proper Tukey loss optimization.
    It is sqrt(influence(x)/x).
    In the context of factor graph optimization, this means weighting both the
    residual and the design matrix by w(e).

    Tukey influence function:
    phi(e) = e * (1 - (e/k)^2)^2 for |e| ≤ k, else phi(e) = 0

    Tukey sqrt-weight function: w(e) = sqrt(phi(e)/e) 
        = (1-(x/k)**2) for |e| ≤ k, else w(e) = 0

    Inputs:
    - error: A float or a list/array of floats representing error(s).
    - k: The scale parameter for the Tukey function, default is 1.0.

    Returns:
    - A float or NumPy array of weights that give proper Tukey cost with (w*e)^2.

    """
    error = np.asarray(error)  # converts float, list, or array into a NumPy array

    abs_error = np.abs(error)
    
    # For |e| <= k: compute weight based on Tukey formula
    # For |e| > k: weight is 0
    weights = np.where(
        abs_error <= k,
        (1 - (error / k)**2),  # Weight for small errors
        0.0  # Weight is zero for large errors
    )

    # Return scalar if input was scalar
    if np.isscalar(error) or np.ndim(error) == 0:
        return float(weights)
    else:
        return weights
    
def l2_cost_function(error: Union[float, list, np.ndarray]) -> Union[float, np.ndarray]:
    """Standard L2 loss function.
    rho(x) = x^2/2.

    Inputs:
    - error: A float or a list/array of floats representing error(s).

    Returns:
    - A float or NumPy array of L2 loss values.

    """
    error = np.asarray(error)  # converts float, list, or array into a NumPy array

    loss = error**2/2

    # Return scalar if input was scalar
    if np.isscalar(error) or np.ndim(error) == 0:
        return float(loss)
    else:
        return loss
    
def l2_influence(error: Union[float, list, np.ndarray]) -> Union[float, np.ndarray]:
    """Standard L2 influence function (derivative of the L2 loss).

    Influence function: phi(e) = d(rho)/d(e) = e

    Inputs:
    - error: A float or a list/array of floats representing error(s).

    Returns:
    - A float or NumPy array of influence values.

    """
    return error  # Directly return error as influence

def l2_sqrt_weight(error: Union[float, list, np.ndarray], **kwargs) -> Union[float, np.ndarray]:
    """Standard L2 sqrt-weight function.

    This implementation gives proper L2 cost optimization when used as cost = (w(e) * e)^2.
    In the context of factor graph optimization, this means weighting both the
    residual and the design matrix by w(e).

    L2 influence function: phi(e) = e
    Sqrt-weight function: w(e) = phi(e)/e = 1.0

    To achieve this with (w*e)^2, we need:
    w = 1 for all e

    Inputs:
    - error: A float or a list/array of floats representing error(s).

    Returns:
    - A float or NumPy array of weights that give proper L2 cost with (w*e)^2.

    """
    error = np.asarray(error)  # converts float, list, or array into a NumPy array

    weights = np.ones_like(error)

    # Return scalar if input was scalar
    if np.isscalar(error) or np.ndim(error) == 0:
        return float(weights)
    else:
        return weights
    
def gen_partial_robust_weight_function(weight_function, *args, **kwargs) -> Callable:
    """Generate a partial function for the robust weight function with given arguments."""
    return partial(weight_function, *args, **kwargs)


def plot_losses_influences_weights():
    """For L2, Huber, DCS, Cauchy, and Tukey, plot the loss functions, influence functions, and weight functions over the range -3 to 3."""
    import matplotlib.pyplot as plt
    x = np.linspace(-3, 3, 500)
    k = 1.0  # Common scale parameter for robust functions
    functions = {
        'L2': (l2_cost_function, l2_influence, l2_sqrt_weight),
        'Huber': (partial(huber_loss_function, k=k), partial(huber_influence, k=k), partial(huber_sqrt_weight, k=k)),
        'DCS': (partial(dcs_loss_function, k=k), partial(dcs_influence, k=k), partial(dcs_sqrt_weight, k=k)),
        'Cauchy': (partial(cauchy_loss_function, k=k), partial(cauchy_influence, k=k), partial(cauchy_sqrt_weight, k=k)),
        'Tukey': (partial(tukey_loss_function, k=k), partial(tukey_influence, k=k), partial(tukey_sqrt_weight, k=k)),
    }
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))
    for name, (loss_fn, influence_fn, weight_fn) in functions.items():
        axs[0].plot(x, loss_fn(x), label=name)
        axs[1].plot(x, influence_fn(x), label=name)
        axs[2].plot(x, weight_fn(x), label=name)
    axs[0].set_title('Loss Functions')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].grid()
    axs[0].set_xlim([-3, 3])
    axs[1].set_title('Influence Functions')
    axs[1].set_ylabel('Influence')
    axs[1].legend()
    axs[1].grid()
    axs[1].set_xlim([-3, 3])
    axs[2].set_title('Weight Functions')
    axs[2].set_ylabel('Weight')
    axs[2].legend()
    axs[2].grid()
    axs[2].set_xlim([-3, 3])
    plt.xlabel('Error')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_losses_influences_weights()



