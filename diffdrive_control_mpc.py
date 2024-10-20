"""
Example diffdrive_control_mpc.py
Author: Joshua A. Marshall <joshua.marshall@queensu.ca>
GitHub: https://github.com/botprof/agv-examples
"""

# %%
# SIMULATION SETUP

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from mobotpy.models import DiffDrive
from mobotpy.integration import rk_four

# Set the simulation time [s] and the sample period [s]
SIM_TIME = 30.0
T = 0.1

# Create an array of time values [s]
t = np.arange(0.0, SIM_TIME, T)
N = np.size(t)

# %%
# UNCONSTRAINED MPC CONTROLLER DESIGN

# Lookahead time steps
P = 50

# Decide on state and input cost matrices
smallQ = np.diag([1.0, 1.0, 2.0])
smallR = np.diag([1.0, 1.0])

# Create a new desired trajectory time array with sufficient time for the MPC
t_d = np.arange(0.0, SIM_TIME + P * T, T)

# %%
# VEHICLE SETUP

# Set the track length of the vehicle [m]
ELL = 1.0

# Create a vehicle object of type DiffDrive
vehicle = DiffDrive(ELL)

# %%
# COMPUTE THE REFERENCE TRAJECTORY

# Radius of the circle [m]
R = 10.0

# Angular rate [rad/s] at which to traverse the circle
OMEGA = 0.1

# Pre-compute the desired trajectory
x_d = np.zeros((3, N + P))
u_d = np.zeros((2, N + P))
for k in range(0, int(N / 2)):
    x_d[0, k] = R * np.sin(OMEGA * t_d[k])
    x_d[1, k] = R * (1 - np.cos(OMEGA * t_d[k]))
    x_d[2, k] = OMEGA * t_d[k]
    u_d[:, k] = vehicle.uni2diff(np.array([R * OMEGA, OMEGA]))

for k in range(int(N / 2), N + P):
    x_d[0, k] = x_d[0, k - 1] + R * OMEGA * T
    x_d[1, k] = x_d[1, k - 1]
    x_d[2, k] = 0
    u_d[:, k] = vehicle.uni2diff(np.array([R * OMEGA, 0]))

# %%
# SIMULATE THE CLOSED-LOOP SYSTEM

# Initial conditions
x_init = np.zeros(3)
x_init[0] = 0.0
x_init[1] = 3.0
x_init[2] = 0.0

# Setup some arrays
x = np.zeros((3, N))
u = np.zeros((2, N))
x[:, 0] = x_init

for k in range(1, N):

    # Simulate the differential drive vehicle motion
    x[:, k] = rk_four(vehicle.f, x[:, k - 1], u[:, k - 1], T)

    # Set vectors for optimization
    x_MPC = cp.Variable((3, P))
    u_MPC = cp.Variable((2, P))

    # Initialize the cost function and constraints
    J = 0
    constraints = []

    # For each lookahead step
    for j in range(0, P):

        # Compute the approximate linearization
        F = np.array(
            [
                [
                    1,
                    0,
                    -T * 0.5
                    * (u_d[0, k + j - 1] + u_d[1, k + j - 1])
                    * np.sin(x_d[2, k + j - 1]),
                ],
                [
                    0,
                    1,
                    T * 0.5
                    * (u_d[0, k + j - 1] + u_d[1, k + j - 1])
                    * np.cos(x_d[2, k + j - 1]),
                ],
                [0, 0, 1],
            ]
        )
        G = T * np.array(
            [
                [0.5 * np.cos(x_d[2, k + j - 1]), 0.5 * np.cos(x_d[2, k + j - 1])],
                [0.5 * np.sin(x_d[2, k + j - 1]), 0.5 * np.sin(x_d[2, k + j - 1])],
                [-1 / ELL, 1 / ELL],
            ]
        )

        # Increment the cost function
        J += cp.quad_form(x_MPC[:, j] - x_d[:, k + j], smallQ) + cp.quad_form(
            u_MPC[:, j], smallR
        )

        # Enter the "subject to" constraints
        constraints += [
            x_MPC[:, j]
            == x_d[:, k + j]
            + F @ (x_MPC[:, j - 1] - x_d[:, k + j - 1])
            + G @ (u_MPC[:, j - 1] - u_d[:, k + j - 1])
        ]
        constraints += [x_MPC[:, 0] == x[:, k]]
        # constraints += [u_MPC[:, j] <= 1.5 * np.ones(2)]
        # constraints += [u_MPC[:, j] >= -1.5 * np.ones(2)]

    # Solve the optimization problem
    problem = cp.Problem(cp.Minimize(J), constraints)
    problem.solve(verbose=False)

    # Set the control input to the first element of the solution
    u[:, k] = u_MPC[:, 0].value

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
plt.plot(t, x_d[0, 0:N], "C1--")
plt.plot(t, x[0, :], "C0")
plt.grid(color="0.95")
plt.ylabel(r"$x$ [m]")
plt.setp(ax1a, xticklabels=[])
plt.legend(["Desired", "Actual"])
ax1b = plt.subplot(412)
plt.plot(t, x_d[1, 0:N], "C1--")
plt.plot(t, x[1, :], "C0")
plt.grid(color="0.95")
plt.ylabel(r"$y$ [m]")
plt.setp(ax1b, xticklabels=[])
ax1c = plt.subplot(413)
plt.plot(t, x_d[2, 0:N] * 180.0 / np.pi, "C1--")
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
# plt.savefig("../agv-book/figs/ch4/control_approx_linearization_fig1.pdf")

# Plot the position of the vehicle in the plane
fig2 = plt.figure(2)
plt.plot(x_d[0, 0:N], x_d[1, 0:N], "C1--", label="Desired")
plt.plot(x[0, :], x[1, :], "C0", label="Actual")
plt.axis("equal")
X_L, Y_L, X_R, Y_R, X_B, Y_B, X_C, Y_C = vehicle.draw(x[0, 0], x[1, 0], x[2, 0])
plt.fill(X_L, Y_L, "k")
plt.fill(X_R, Y_R, "k")
plt.fill(X_C, Y_C, "k")
plt.fill(X_B, Y_B, "C2", alpha=0.5, label="Start")
X_L, Y_L, X_R, Y_R, X_B, Y_B, X_C, Y_C = vehicle.draw(
    x[0, N - 1], x[1, N - 1], x[2, N - 1]
)
plt.fill(X_L, Y_L, "k")
plt.fill(X_R, Y_R, "k")
plt.fill(X_C, Y_C, "k")
plt.fill(X_B, Y_B, "C3", alpha=0.5, label="End")
plt.xlabel(r"$x$ [m]")
plt.ylabel(r"$y$ [m]")
plt.legend()

# Save the plot
# plt.savefig("../agv-book/figs/ch4/control_approx_linearization_fig2.pdf")

# Show all the plots to the screen
plt.show()

# %%
# MAKE AN ANIMATION

# Create the animation
ani = vehicle.animate_trajectory(x, x_d, T)

# Create and save the animation
# ani = vehicle.animate_trajectory(
#     x, x_d, T, True, "../agv-book/gifs/ch4/control_approx_linearization.gif"
# )

# Show the movie to the screen
plt.show()

# Show animation in HTML output if you are using IPython or Jupyter notebooks
# from IPython.display import display

# plt.rc("animation", html="jshtml")
# display(ani)
# plt.close()
