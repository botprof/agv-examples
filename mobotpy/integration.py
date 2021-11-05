"""
Python module integration.py for numerical integration routines.
Author: Joshua A. Marshall <joshua.marshall@queensu.ca>
GitHub: https://github.com/botprof/agv-examples
"""


def rk_four(f, x, u, T, params):
    """
    Perform fourth-order Runge-Kutta numerical integration.

    The function to integrate is f(x, u, params), where the state varables are 
    collected in the variable x, we assume a constant input vector u over time 
    interval T, and params is an array of the system's parameters.
    """
    k_1 = f(x, u, params)
    k_2 = f(x+k_1/2.0, u, params)
    k_3 = f(x+k_2/2.0, u, params)
    k_4 = f(x+k_3, u, params)
    x_new = x+T/6.0*(k_1+2.0*k_2+2.0*k_3+k_4)
    return x_new


def euler_int(f, x, u, T, params):
    """
    Perform Euler (trapezoidal) numerical integration.

    The function to integrate is f(x, u, params), where the state varables are 
    collected in the variable x, we assume a constant input vector u over time 
    interval T, and params is an array of the system's parameters.
    """
    x_new = x+T*f(x, u, params)
    return x_new
