"""
Example dynamic_extension_tracking.py
Author: Joshua A. Marshall <joshua.marshall@queensu.ca>
GitHub: https://github.com/botprof/agv-examples
"""

# %%
# SIMULATION SETUP

import numpy as np
import matplotlib.pyplot as plt
from mobotpy import models
from mobotpy import integration
from scipy import signal

# Set the simulation time [s] and the sample period [s]
SIM_TIME = 15.0
T = 0.04

# Create an array of time values [s]
t = np.arange(0.0, SIM_TIME, T)
N = np.size(t)

# %%
# COMPUTE THE REFERENCE TRAJECTORY

# Radius of the circle [m]
R = 10

# Angular rate [rad/s] at which to traverse the circle
OMEGA = 0.1

# Precompute the desired trajectory
x_d = np.zeros((3, N))
u_d = np.zeros((2, N))
z_d = np.zeros((4, N))
for k in range(0, N):
    x_d[0, k] = R * np.sin(OMEGA * t[k])
    x_d[1, k] = R * (1 - np.cos(OMEGA * t[k]))
    x_d[2, k] = OMEGA * t[k]
    u_d[0, k] = R * OMEGA
    u_d[1, k] = OMEGA

# Precompute the extended system reference trajectory
for k in range(0, N):
    z_d[0, k] = x_d[0, k]
    z_d[1, k] = x_d[1, k]
    z_d[2, k] = u_d[0, k] * np.cos(x_d[2, k])
    z_d[3, k] = u_d[0, k] * np.sin(x_d[2, k])

# %%
# VEHICLE SETUP

# Set the track length of the vehicle [m]
ELL = 1.0

# Create a vehicle object of type DiffDrive
vehicle = models.DiffDrive(ELL)

# %%
# SIMULATE THE CLOSED-LOOP SYSTEM

# Initial conditions
x_init = np.zeros(3)
x_init[0] = 0.0
x_init[1] = 3.0
x_init[2] = 0.0

# Setup some arrays
x = np.zeros((3, N))
z = np.zeros((4, N))
u = np.zeros((2, N))
x[:, 0] = x_init

# Set the initial speed [m/s] to be non-zero to avoid singularity
w = np.zeros(2)
u_unicycle = np.zeros(2)
u_unicycle[0] = u_d[0, 0]

# Initial extended state
z[0, 0] = x_init[0]
z[1, 0] = x_init[1]
z[2, 0] = u_d[0, 0] * np.cos(x_init[2])
z[3, 0] = u_d[0, 0] * np.sin(x_init[2])

# Defined feedback linearized state matrices
A = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
B = np.array([[0, 0], [0, 0], [1, 0], [0, 1]])

# Choose pole locations for closed-loop linear system
p = np.array([-1.0, -2.0, -2.5, -1.5])
K = signal.place_poles(A, B, p)

for k in range(1, N):

    # Compute the extended linear system input control signals
    eta = K.gain_matrix @ (z_d[:, k - 1] - z[:, k - 1])

    # Compute the new (unicycle) vehicle inputs
    B_inv = np.array(
        [
            [np.cos(x[2, k - 1]), np.sin(x[2, k - 1])],
            [-np.sin(x[2, k - 1]) / u_unicycle[0], np.cos(x[2, k - 1]) / u_unicycle[0]],
        ]
    )
    w = B_inv @ eta
    u_unicycle[0] = u_unicycle[0] + T * w[0]
    u_unicycle[1] = w[1]

    # Convert unicycle inputs to differential drive wheel speeds
    u[:, k] = vehicle.uni2diff(u_unicycle, ELL)

    # Simulate the vehicle motion
    x[:, k] = integration.rk_four(vehicle.f, x[:, k - 1], u[:, k - 1], T, ELL)

    # Update the extended system states
    z[0, k] = x[0, k]
    z[1, k] = x[1, k]
    z[2, k] = u_unicycle[0] * np.cos(x[2, k])
    z[3, k] = u_unicycle[0] * np.sin(x[2, k])

# %%
# MAKE PLOTS

# Change some plot settings (optional)
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{cmbright,amsmath,bm}")
plt.rc("savefig", format="pdf")
plt.rc("savefig", bbox="tight")

# Plot the states as a function of time
fig1 = plt.figure(1)
fig1.set_figheight(6.4)
ax1a = plt.subplot(411)
plt.plot(t, x_d[0, :], "C1--")
plt.plot(t, x[0, :], "C0")
plt.grid(color="0.95")
plt.ylabel(r"$x$ [m]")
plt.setp(ax1a, xticklabels=[])
plt.legend(["Desired", "Actual"])
ax1b = plt.subplot(412)
plt.plot(t, x_d[1, :], "C1--")
plt.plot(t, x[1, :], "C0")
plt.grid(color="0.95")
plt.ylabel(r"$y$ [m]")
plt.setp(ax1b, xticklabels=[])
ax1c = plt.subplot(413)
plt.plot(t, x_d[2, :] * 180.0 / np.pi, "C1--")
plt.plot(t, x[2, :] * 180.0 / np.pi, "C0")
plt.grid(color="0.95")
plt.ylabel(r"$\theta$ [deg]")
plt.setp(ax1c, xticklabels=[])
ax1d = plt.subplot(414)
plt.step(t, u[0, :], "C2", where="post", label="$v_L$")
plt.step(t, u[1, :], "C3", where="post", label="$v_R$")
plt.grid(color="0.95")
plt.ylabel(r"$\bm{u}$ [m/s]")
plt.xlabel(r"$t$ [s]")
plt.legend()

# Save the plot
plt.savefig("../agv-book/figs/ch4/dynamic_extension_tracking_fig1.pdf")

# Plot the position of the vehicle in the plane
fig2 = plt.figure(2)
plt.plot(x_d[0, :], x_d[1, :], "C1--", label="Desired")
plt.plot(x[0, :], x[1, :], "C0", label="Actual")
plt.axis("equal")
X_L, Y_L, X_R, Y_R, X_B, Y_B, X_C, Y_C = vehicle.draw(x[0, 0], x[1, 0], x[2, 0], ELL)
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
plt.savefig("../agv-book/figs/ch4/dynamic_extension_tracking_fig2.pdf")

# Show all the plots to the screen
plt.show()

# %%
# MAKE AN ANIMATION

# Create and save the animation
ani = vehicle.animate_trajectory(
    x, x_d, T, ELL, True, "../agv-book/gifs/ch4/dynamic_extension_tracking.gif"
)

# Show the movie to the screen
plt.show()

# # Show animation in HTML output if you are using IPython or Jupyter notebooks
# plt.rc('animation', html='jshtml')
# display(ani)
# plt.close()
