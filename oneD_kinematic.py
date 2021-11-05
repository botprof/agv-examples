"""
Example oneD_kinematic.py
Author: Joshua A. Marshall <joshua.marshall@queensu.ca>
GitHub: https://github.com/botprof/agv-examples
"""

# %% SIMULATION SETUP

import matplotlib.pyplot as plt
import numpy as np
from mobotpy import models

# Set the simulation time [s] and the sample period [s]
SIM_TIME = 30.0
T = 0.4

# Create an array of time values [s]
t = np.arange(0, SIM_TIME, T)
N = np.size(t)

# %% FUNCTION DEFNITIONS

def vehicle(x, u, T):
    """Discrete-time 1D kinematic vehicle model."""
    x_new = x+T*u
    return x_new

# %% RUN SIMULATION

# Initialize arrays that will be populated with our inputs and states
x = np.zeros(N)
u = np.zeros(N)

# Set the initial position [m] and input [m/s] and run the simulation
x[0] = 1.0
u[0] = 0.0
for k in range(1, N):
    x[k] = vehicle(x[k-1], u[k-1], T)
    u[k] = np.sin(k*T)

# %% MAKE A PLOT

# Change some plot settings (optional)
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{cmbright,amsmath}')
plt.rc('savefig', format='pdf')

# Plot the state (x) and input (u) vs time (t)
fig1 = plt.figure(1)
ax1a = plt.subplot(211)
plt.plot(t, x, 'C0')
plt.grid(color='0.95')
plt.ylabel(r'$x$ [m]')
plt.setp(ax1a, xticklabels=[])
ax1b = plt.subplot(212)
plt.step(t, u, 'C1', where='post')
plt.grid(color='0.95')
plt.ylabel(r'$u$ [m/s]')
plt.xlabel(r'$t$ [s]')

# Save the plot
plt.savefig('../agv-book/figs/ch2/oneD_kinematic_fig1.pdf')

# %% MAKE AN ANIMATION

# Set the side length of the vehicle [m]
LENGTH = 1.0

# Let's use the Cart class to create an animation
vehicle = models.Cart(LENGTH)

# Create and save the animation
ani = vehicle.animate(x, T, LENGTH, True,
                      '../agv-book/gifs/ch2/oneD_kinematic.gif')

# %% DISPLAY PLOTS

# Show all the plots to the screen
plt.show()
