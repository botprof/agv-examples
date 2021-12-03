"""
Example diffdrive_kinematic.py
Author: Joshua A. Marshall <joshua.marshall@queensu.ca>
GitHub: https://github.com/botprof/agv-examples
"""

# %% SIMULATION SETUP

import numpy as np
import matplotlib.pyplot as plt
from mobotpy import models

# Set the simulation time [s] and the sample period [s]
SIM_TIME = 30.0
T = 0.04

# Create an array of time values [s]
t = np.arange(0.0, SIM_TIME, T)
N = np.size(t)

# %% FUNCTION DEFINITIONS

def rk_four(f, x, u, T, P):
    """Fourth-order Runge-Kutta numerical integration."""
    k_1 = f(x, u, P)
    k_2 = f(x+T*k_1/2.0, u, P)
    k_3 = f(x+T*k_2/2.0, u, P)
    k_4 = f(x+T*k_3, u, P)
    x_new = x+T/6.0*(k_1+2.0*k_2+2.0*k_3+k_4)
    return x_new

def diffdrive_f(x, u, ell):
    """Differential drive kinematic vehicle model."""
    f = np.zeros(3)
    f[0] = 0.5*(u[0]+u[1])*np.cos(x[2])
    f[1] = 0.5*(u[0]+u[1])*np.sin(x[2])
    f[2] = 1.0/ell*(u[1]-u[0])
    return f

def uni2diff(u, ell):
    '''Convert speed and angular rate to wheel speeds.'''
    v = u[0]
    omega = u[1]
    v_L = v-ell/2*omega
    v_R = v+ell/2*omega
    return np.array([v_L, v_R])

def openloop(t):
    """Specify open loop speed and angular rate inputs."""
    v = 0.5
    omega = 0.5*np.sin(10*t*np.pi/180.0)
    return np.array([v, omega])

# %% RUN SIMULATION

# Initialize arrays that will be populated with our inputs and states
x = np.zeros((3, N))
u = np.zeros((2, N))

# Set the track of the vehicle [m]
ELL = 0.35

# Set the initial pose [m, m, rad], velocities [m/s, m/s]
x[0, 0] = 0.0
x[1, 0] = 0.0
x[2, 0] = np.pi/2.0
u[:, 0] = uni2diff(openloop(t[0]), ELL)

# Run the simulation
for k in range(1, N):
    x[:, k] = rk_four(diffdrive_f, x[:, k-1], u[:, k-1], T,
                      ELL)
    u[:, k] = uni2diff(openloop(t[k]), ELL)

# %% MAKE PLOTS

# Change some plot settings (optional)
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{cmbright,amsmath,bm}')
plt.rc('savefig', format='pdf')

# Plot the states as a function of time
fig1 = plt.figure(1)
fig1.set_figheight(6.4)
ax1a = plt.subplot(411)
plt.plot(t, x[0, :])
plt.grid(color='0.95')
plt.ylabel(r'$x$ [m]')
plt.setp(ax1a, xticklabels=[])
ax1b = plt.subplot(412)
plt.plot(t, x[1, :])
plt.grid(color='0.95')
plt.ylabel(r'$y$ [m]')
plt.setp(ax1b, xticklabels=[])
ax1c = plt.subplot(413)
plt.plot(t, x[2, :]*180.0/np.pi)
plt.grid(color='0.95')
plt.ylabel(r'$\theta$ [deg]')
plt.setp(ax1c, xticklabels=[])
ax1d = plt.subplot(414)
plt.step(t, u[0, :], 'C1', where='post', label='$v_L$')
plt.step(t, u[1, :], 'C2', where='post', label='$v_R$')
plt.grid(color='0.95')
plt.ylabel(r'$\bm{u}$ [m/s]')
plt.xlabel(r'$t$ [s]')
plt.legend()

# Save the plot
plt.savefig('../agv-book/figs/ch3/diffdrive_kinematic_fig1.pdf')

# Let's now use the class DiffDrive for plotting
vehicle = models.DiffDrive(ELL)

# Plot the position of the vehicle in the plane
fig2 = plt.figure(2)
plt.plot(x[0, :], x[1, :], 'C0')
plt.axis('equal')
X_L, Y_L, X_R, Y_R, X_B, Y_B, X_C, Y_C = vehicle.draw(x[0, 0], x[1, 0],
                                                     x[2, 0], ELL)
plt.fill(X_L, Y_L, 'k')
plt.fill(X_R, Y_R, 'k')
plt.fill(X_C, Y_C, 'k')
plt.fill(X_B, Y_B, 'C0', alpha=0.5, label='Start')
X_L, Y_L, X_R, Y_R, X_B, Y_B, X_C, Y_C =vehicle.draw(x[0, N-1], x[1, N-1],
                                                     x[2, N-1], ELL)
plt.fill(X_L, Y_L, 'k')
plt.fill(X_R, Y_R, 'k')
plt.fill(X_C, Y_C, 'k')
plt.fill(X_B, Y_B, 'C1', alpha=0.5, label='End')
plt.xlabel(r'$x$ [m]')
plt.ylabel(r'$y$ [m]')
plt.legend()

# Save the plot
plt.savefig('../agv-book/figs/ch3/diffdrive_kinematic_fig2.pdf')

# %% MAKE AN ANIMATION

# Create and save the animation
ani = vehicle.animate(x, T, ELL, True,
                                 '../agv-book/gifs/ch3/diffdrive_kinematic.gif')

# %%

# Show all the plots to the screen
plt.show()
