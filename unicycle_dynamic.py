"""
Example unicycle_dynamic.py
Author: Joshua A. Marshall <joshua.marshall@queensu.ca>
GitHub: https://github.com/botprof/agv-examples
"""

# %% SIMULATION SETUP

import numpy as np
import matplotlib.pyplot as plt
from mobotpy import integration
from mobotpy import models

# Set the simulation time [s] and the sample period [s]
SIM_TIME = 60.0
T = 0.04

# Create an array of time values [s]
t = np.arange(0.0, SIM_TIME, T)
N = np.size(t)

# Set the vehicle's mass and moment of inertia
m = 1.0  # kg
I = 1.0  # kg m^2

# Set the maximum lateral tire force [N]
lambda_max = 0.1

# %% FUNCTION DEFINITIONS


def unicycle_f_dyn(x, u, params):
    """Unicycle dynamic vehicle model.
    
    Parameters
    ----------
    x : ndarray of length 6
        The vehicle's state (x, y, theta, v_x, v_y, v_2).
    u : ndarray of length 2
        Force and torque applied to the wheel (f, tau).
    params : ndarray of length 3
        The mass [kg], moment of inertia [kg m^2] (m, I), and lateral tire 
        force lambda_f [N].

    Returns
    -------
    f_dyn : ndarray of length 6
        The rate of change of the vehicle states.
    """
    m = params[0]
    I = params[1]
    lambda_f = params[2]
    f_dyn = np.zeros(6)
    f_dyn[0] = x[3]
    f_dyn[1] = x[4]
    f_dyn[2] = x[5]
    f_dyn[3] = 1.0/m*(u[0]*np.cos(x[2]) - lambda_f*np.sin(x[2]))
    f_dyn[4] = 1.0/m*(u[0]*np.sin(x[2]) + lambda_f*np.cos(x[2]))
    f_dyn[5] = 1.0/I*u[1]
    return f_dyn


def lateral_force(x, lambda_max, params):
    """Computes the lateral tire force for a single wheel.
    
    Parameters
    ----------
    x : ndarray of length 6
        The vehicle's state (x, y, theta, v_x, v_y, v_2).
    lambda_max : float
        Maximum lateral tire force [N] the tire will support.
    params : ndarray of length 3
        The mass [kg], moment of inertia [kg m^2] (m, I).

    Returns
    -------
    lambda_f : float
        The computed lateral tire force [N].
    old_x : ndarray of length 6
        The vehicle's state with or without slip.
    """
    m = params[0]
    I = params[1]

    # Compute lateral force
    lambda_f = m*x[5]*(x[3]*np.cos(x[2])+x[4]*np.sin(x[2]))

    # Check whether the required lateral force is bigger than the tire can
    # handle
    if np.abs(lambda_f) > lambda_max:
        # Wheel slips
        lambda_f = lambda_max*np.sign(lambda_f)
        old_vx = x[3]
        old_vy = x[4]
    else:
        # Wheel doesn't slip, so enforce velocity to be in the direction of the
        # wheel
        old_vx = x[3]*np.square(np.cos(x[2]))+x[4]*np.sin(x[2])*np.cos(x[2])
        old_vy = x[4]*np.square(np.sin(x[2]))+x[3]*np.sin(x[2])*np.cos(x[2])

    # Assign the new state
    old_x = np.array([x[0], x[1], x[2], old_vx, old_vy, x[5]])

    # Return the output
    return lambda_f, old_x

# %% RUN SIMULATION


# Initialize arrays that will be populated with our inputs and states
x = np.zeros((6, N))
u = np.zeros((2, N))
lambda_f = np.zeros((1, N))

# Set the initial conditions
x_init = np.zeros(6)
x_init[0] = 0.0
x_init[1] = 3.0
x_init[2] = 0.0
x_init[3] = 0.0
x_init[4] = 0.0
x_init[5] = 0.0
x[:, 0] = x_init

# Run the simulation
for k in range(1, N):

    # Make some force and torque inputs to steer the vehicle around
    if k < round(N/6):
        u[0, k-1] = 0.1
        u[1, k-1] = 0.0
    elif k < round(3*N/6):
        u[0, k-1] = 0.0
        u[1, k-1] = -0.01
    elif k < round(4*N/6):
        u[0, k-1] = 0.0
        u[1, k-1] = 0.03
    else:
        u[0, k-1] = 0.0
        u[1, k-1] = 0.0

    # Compute the lateral force applied to the vehicle's wheel
    lambda_f[0, k-1], x[:, k-1] = lateral_force(x[:, k-1], lambda_max,
                                                np.array([m, I]))

    # Update the motion of the vehicle
    x[:, k] = integration.rk_four(unicycle_f_dyn, x[:, k-1], u[:, k-1], T,
                                  np.array([m, I, lambda_f[0, k-1]]))

# %% MAKE PLOTS

# Change some plot settings (optional)
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{cmbright,amsmath,bm}')
plt.rc('savefig', format='pdf')

# Plot the states as a function of time
fig1 = plt.figure(1)
ax1 = plt.subplot(311)
plt.setp(ax1, xticklabels=[])
plt.plot(t, x[0, :], 'C0')
plt.ylabel(r'$x$ [m]')
plt.grid(color='0.95')
ax2 = plt.subplot(312)
plt.setp(ax2, xticklabels=[])
plt.plot(t, x[1, :], 'C0')
plt.ylabel(r'$y$ [m]')
plt.grid(color='0.95')
ax3 = plt.subplot(313)
plt.plot(t, x[2, :], 'C0')
plt.xlabel(r'$t$ [s]')
plt.ylabel(r'$\theta$ [rad]')
plt.grid(color='0.95')

# Save the plot
plt.savefig('../agv-book/figs/ch3/unicycle_dynamic_fig1.pdf')

# Plot the lateral tire force
fig2 = plt.figure(2)
plt.plot(t[0:N-1], lambda_f[0, 0:N-1], 'C0')
plt.xlabel(r'$t$ [s]')
plt.ylabel(r'$\lambda$ [N]')
plt.grid(color='0.95')

# Save the plot
plt.savefig('../agv-book/figs/ch3/unicycle_dynamic_fig2.pdf')

'''
To keep thing simple, we plot the unicycle as a differential drive vehicle,
because the differential drive vehicle has the same nonholonomic constraints.
'''

# Set the track of the vehicle [m]
ELL = 1.0

# Let's now use the class DiffDrive for plotting
vehicle = models.DiffDrive(ELL)

# Plot the position of the vehicle in the plane
fig3 = plt.figure(3)
plt.plot(x[0, :], x[1, :], 'C0')
plt.axis('equal')
X_L, Y_L, X_R, Y_R, X_B, Y_B, X_C, Y_C = vehicle.draw(x[0, 0], x[1, 0],
                                                      x[2, 0], ELL)
plt.fill(X_L, Y_L, 'k')
plt.fill(X_R, Y_R, 'k')
plt.fill(X_C, Y_C, 'k')
plt.fill(X_B, Y_B, 'C0', alpha=0.5, label='Start')
X_L, Y_L, X_R, Y_R, X_B, Y_B, X_C, Y_C = vehicle.draw(x[0, N-1], x[1, N-1],
                                                      x[2, N-1], ELL)
plt.fill(X_L, Y_L, 'k')
plt.fill(X_R, Y_R, 'k')
plt.fill(X_C, Y_C, 'k')
plt.fill(X_B, Y_B, 'C1', alpha=0.5, label='End')
plt.xlabel(r'$x$ [m]')
plt.ylabel(r'$y$ [m]')
plt.legend()

# Save the plot
plt.savefig('../agv-book/figs/ch3/unicycle_dynamic_fig3.pdf')

# %% MAKE AN ANIMATION

# Create and save the animation
ani = vehicle.animate(x, T, ELL, True,
                      '../agv-book/gifs/ch3/unicycle_dynamic.gif')

# %%

# Show the plots
plt.show()
