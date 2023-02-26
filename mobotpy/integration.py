"""
Python module integration.py for numerical integration routines.
Author: Joshua A. Marshall <joshua.marshall@queensu.ca>
GitHub: https://github.com/botprof/agv-examples
"""


def rk_four(f, x, u, T):
    """
    Perform fourth-order Runge-Kutta numerical integration.

    The function to integrate is f(x, u, params), where the state variables are
    collected in the variable x, we assume a constant input vector u over time
    interval T > 0, and params is an array of the system's parameters.
    """
    k_1 = f(x, u)
    k_2 = f(x + T * k_1 / 2.0, u)
    k_3 = f(x + T * k_2 / 2.0, u)
    k_4 = f(x + T * k_3, u)
    x_new = x + T / 6.0 * (k_1 + 2.0 * k_2 + 2.0 * k_3 + k_4)
    return x_new


def euler_int(f, x, u, T):
    """
    Perform Euler (trapezoidal) numerical integration.

    The function to integrate is f(x, u, params), where the state variables are
    collected in the variable x, we assume a constant input vector u over time
    interval T > 0, and params is an array of the system's parameters.
    """
    x_new = x + T * f(x, u)
    return x_new
