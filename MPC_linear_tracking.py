"""
Example MPC_linear_tracking.py
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
from numpy.linalg import matrix_power
from numpy.linalg import inv

# Set the simulation time [s] and the sample period [s]
SIM_TIME = 30.0
T = 0.01

# Create an array of time values [s]
t = np.arange(0.0, SIM_TIME, T)
N = np.size(t)

# %%
# VEHICLE MODELS

# Vehicle mass [kg]
M = 1.0

# Discrete time vehicle model
F = np.array([[1, T], [0, 1]])
G = np.array([[0], [T / M]])
n = G.shape[0]

# Continuous time vehicle model (for full-state feedback)
A = np.array([[0, 1], [0, 0]])
B = np.array([[0], [1 / M]])

# %%
# UNCONSTRAINED MPC CONTROLLER DESIGN

# Lookahead time steps
p = 1000

# Construct the matrices M and L
L = np.zeros((n * p, n))
M = np.zeros((n * p, p))
for i in range(0, p):
    L[n * i : n * i + n, 0:n] = matrix_power(F, i + 1)
    for j in range(0, p - i):
        M[n * (p - i) - n : n * (p - i), j : j + 1] = matrix_power(F, p - i - j - 1) @ G

# Decide on state and input cost matrices
smallQ = np.array([[1.0, 0.0], [0.0, 1.0]])
Q = 1.0 * np.kron(np.eye(p), smallQ)
R = 0.1 * np.eye(p)

# Because the system is LTI we can pre-compute the gain matrix
K_MPC = inv(M.T @ Q @ M + R) @ M.T @ Q

# %%
# FULL-STATE FEEDBACK CONTROLLER DESIGN

# Choose some poles for the FSF controller
poles = np.array([-1.0, -2.0])

# Find the controller gain to place the poles at our chosen locations
K_FSF = signal.place_poles(A, B, poles)

# %%
# SIMULATE THE CLOSED-LOOP SYSTEMS

# Set the desired trajectory to take a step mid way through the trajectory
x_d = np.zeros((2, N + p))
for k in range(int(N / 2), N + p):
    x_d[0, k] = 10
    x_d[1, k] = 0

# Set up some more arrays for MPC and FSF
x_MPC = np.zeros((n, N))
u_MPC = np.zeros((1, N))
xi_d = np.zeros(n * p)
x_FSF = np.zeros((n, N))
u_FSF = np.zeros((1, N))

# Set the initial conditions
x_MPC[0, 0] = 3
x_MPC[1, 0] = 0
x_FSF[0, 0] = x_MPC[0, 0]
x_FSF[1, 0] = x_MPC[1, 0]

# Simulate the the closed-loop system with MPC
for k in range(1, N):
    x_MPC[:, k] = F @ x_MPC[:, k - 1] + G @ u_MPC[:, k - 1]
    for j in range(0, p):
        xi_d[n * j : n * j + n] = x_d[:, k + j]
    u = K_MPC @ (xi_d - L @ x_MPC[:, k - 1])
    u_MPC[:, k] = u[0]

# Simulate the closed-loop system with FSF
for k in range(1, N):
    x_FSF[:, k] = F @ x_FSF[:, k - 1] + G @ u_FSF[:, k - 1]
    u_FSF[:, k] = -K_FSF.gain_matrix @ (x_FSF[:, k - 1] - x_d[:, k - 1])

# %%
# PLOT THE RESULTS

# Change some plot settings (optional)
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{cmbright,amsmath}")
plt.rc("savefig", format="pdf")
plt.rc("savefig", bbox="tight")

fig1 = plt.figure(1)
ax1a = plt.subplot(311)
plt.plot(t, x_d[0, 0:N], "C2--")
plt.plot(t, x_MPC[0, :], "C0")
plt.plot(t, x_FSF[0, :], "C1")
plt.grid(color="0.95")
plt.ylabel(r"$x_1$ [m]")
plt.legend(["Desired", "MPC", "Full-state feedback"], loc="lower right")
plt.setp(ax1a, xticklabels=[])
ax1b = plt.subplot(312)
plt.plot(t, x_d[1, 0:N], "C2--")
plt.plot(t, x_MPC[1, :], "C0")
plt.plot(t, x_FSF[1, :], "C1")
plt.grid(color="0.95")
plt.ylabel(r"$x_2$ [m/s]")
plt.setp(ax1b, xticklabels=[])
ax1c = plt.subplot(313)
plt.plot(t, u_MPC[0, :], "C0")
plt.plot(t, u_FSF[0, :], "C1")
plt.grid(color="0.95")
plt.ylabel(r"$u$ [N]")
plt.xlabel(r"$t$ [s]")

# Save the plot
plt.savefig("../agv-book/figs/ch4/MPC_linear_tracking_fig1.pdf")

# Show the plots
plt.show()
