"""
Example control_approx_linearization.py
Author: Joshua A. Marshall <joshua.marshall@queensu.ca>
GitHub: https://github.com/botprof/agv-examples
"""

# %% SIMULATION SETUP

import numpy as np
import matplotlib.pyplot as plt
from mobotpy import models
from mobotpy import integration
from scipy import signal

# Set the simulation time [s] and the sample period [s]
SIM_TIME = 20.0
T = 0.04

# Create an array of time values [s]
t = np.arange(0.0, SIM_TIME, T)
N = np.size(t)

# %% COMPUTE THE REFERENCE TRAJECTORY

# Radius of the circle [m]
R = 10

# Angular rate [rad/s] at which to traverse the circle
GAMMA = 0.1

# Precompute the desired trajectory
xd = np.zeros((3, N))
ud = np.zeros((2, N))
for k in range(0, N):
    xd[0, k] = R * np.sin(GAMMA * t[k])
    xd[1, k] = R * (1 - np.cos(GAMMA * t[k]))
    xd[2, k] = GAMMA * t[k]
    ud[0, k] = R * GAMMA
    ud[1, k] = GAMMA

# %% VEHICLE SETUP

# Set the track length of the vehicle [m]
ELL = 1.0

# Create a vehicle object of type DiffDrive
vehicle = models.DiffDrive(ELL)

# %% SIMULATE THE CLOSED-LOOP SYSTEM

# Initial conditions
x_init = np.zeros(3)
x_init[0] = 0.0
x_init[1] = 3.0
x_init[2] = 0.0

# Setup some arrays
x = np.zeros((3, N))
u = np.zeros(2)
x[:, 0] = x_init

for k in range(1, N):

    # Compute the approximate linearization
    A = np.array(
        [
            (0, 0, -ud[0, k] * np.sin(xd[2, k])),
            (0, 0, ud[0, k] * np.cos(xd[2, k])),
            (0, 0, 0),
        ]
    )
    B = np.array([(np.cos(xd[2, k]), 0), (np.sin(xd[2, k]), 0), (0, 1)])

    # Compute the gain matrix to place poles of (A-BK) at p
    p = np.array([-1.0, -2.0, -0.5])
    K = signal.place_poles(A, B, p)

    # Compute the control signals
    u_unicycle = K.gain_matrix @ (xd[:, k] - x[:, k - 1]) + ud[:, k]
    u = vehicle.uni2diff(u_unicycle, ELL)

    # Simulate the vehicle motion
    x[:, k] = integration.rk_four(vehicle.f, x[:, k - 1], u, T, ELL)

# %% MAKE PLOTS

# Change some plot settings (optional)
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{cmbright,amsmath,bm}")
plt.rc("savefig", format="pdf")
plt.rc("savefig", bbox="tight")

# Plot the states as a function of time
fig1 = plt.figure(1)
ax1a = plt.subplot(311)
plt.plot(t, xd[0, :], "C1--")
plt.plot(t, x[0, :], "C0")
plt.grid(color="0.95")
plt.ylabel(r"$x$ [m]")
plt.setp(ax1a, xticklabels=[])
plt.legend(["Desired", "Actual"])
ax1b = plt.subplot(312)
plt.plot(t, xd[1, :], "C1--")
plt.plot(t, x[1, :], "C0")
plt.grid(color="0.95")
plt.ylabel(r"$y$ [m]")
plt.setp(ax1b, xticklabels=[])
ax1c = plt.subplot(313)
plt.plot(t, xd[2, :] * 180.0 / np.pi, "C1--")
plt.plot(t, x[2, :] * 180.0 / np.pi, "C0")
plt.grid(color="0.95")
plt.ylabel(r"$\theta$ [deg]")
plt.xlabel(r"$t$ [s]")

# Save the plot
plt.savefig("../agv-book/figs/ch4/control_approx_linearization_fig1.pdf")

# Plot the position of the vehicle in the plane
fig2 = plt.figure(2)
plt.plot(xd[0, :], xd[1, :], "C1--", label="Desired")
plt.plot(x[0, :], x[1, :], "C0", label="Actual")
plt.axis("equal")
X_L, Y_L, X_R, Y_R, X_B, Y_B, X_C, Y_C = vehicle.draw(
    x[0, 0], x[1, 0], x[2, 0], ELL)
plt.fill(X_L, Y_L, "k")
plt.fill(X_R, Y_R, "k")
plt.fill(X_C, Y_C, "k")
plt.fill(X_B, Y_B, "C2", alpha=0.5, label="Start")
X_L, Y_L, X_R, Y_R, X_B, Y_B, X_C, Y_C = vehicle.draw(
    x[0, N - 1], x[1, N - 1], x[2, N - 1], ELL
)
plt.fill(X_L, Y_L, "k")
plt.fill(X_R, Y_R, "k")
plt.fill(X_C, Y_C, "k")
plt.fill(X_B, Y_B, "C3", alpha=0.5, label="End")
plt.xlabel(r"$x$ [m]")
plt.ylabel(r"$y$ [m]")
plt.legend()

# Save the plot
plt.savefig("../agv-book/figs/ch4/control_approx_linearization_fig2.pdf")

# %% MAKE AN ANIMATION

# Create and save the animation
ani = vehicle.animate_trajectory(
    x, xd, T, ELL, True, "../agv-book/gifs/ch4/control_approx_linearization.gif"
)

# %%

# Show all the plots to the screen
plt.show()
