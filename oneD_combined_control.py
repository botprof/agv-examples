"""
Example oneD_combined_control.py
Author: Joshua A. Marshall <joshua.marshall@queensu.ca>
GitHub: https://github.com/botprof/agv-examples
"""

# %% SIMULATION SETUP

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from mobotpy.models import Cart

# Set some variables that describe the desired behaviour
ZETA = 1.1
OMEGA_N = np.sqrt(0.3)

# Define the sample time [s]
T = 0.5

# Define the vehicle mass [kg]
M = 10.0

# Compute the controller pole locations
lambda_sc = np.roots([1, 2 * ZETA * OMEGA_N, OMEGA_N ** 2])
lambda_zc = np.exp(lambda_sc * T)

# %% PARAMETERS

# Function that models the vehicle and sensor(s) in discrete time
F = np.array([[1, T], [0, 1]])
G = np.array([[T ** 2 / (2 * M)], [T / M]])
H = np.array([[1, 0]])

# Find gain matrix K that places the poles at lambda_zc
K = signal.place_poles(F, G, lambda_zc)

# Choose estimator gains for stability
lambda_zo = np.array([0.5, 0.4])
LT = signal.place_poles(F.T, H.T, lambda_zo)

# %% FUNCTION DEFINITIONS


def vehicle(x, u, F, G):
    """Discrete-time 1D dynamic vehicle model."""
    x_new = F @ x + G @ [u]
    return x_new


def controller(x, K):
    """Full state feedback controller."""
    u = -K @ x
    return u


def observer(x_hat, u, y, F, G, H, L):
    """Observer-based state estimator."""
    x_hat_new = F @ x_hat + G @ [u] + L @ ([y] - H @ x_hat)
    return x_hat_new


# %% RUN SIMULATION

# Create an array of time values [s]
SIM_TIME = 20.0
t = np.arange(0, SIM_TIME, T)
N = np.size(t)

# Initialize arrays that will be populated with our inputs and states
x = np.zeros((2, N))
u = np.zeros(N)
x_hat = np.zeros((2, N))

# Set the initial position [m], velocity [m/s], and force input [N]
x[0, 0] = 1.0
x[1, 0] = 0.0
u[0] = 0.0

# Set the initial estimated position (different from the actual position)
x_hat[0, 0] = 0.0
x_hat[0, 0] = 0.0

# Run the simulation
for k in range(1, N):
    y = x[0, k - 1]
    x_hat[:, k] = observer(x_hat[:, k - 1], u[k - 1], y, F, G, H, LT.gain_matrix.T)
    x[:, k] = vehicle(x[:, k - 1], u[k - 1], F, G)
    u[k] = controller(x_hat[:, k], K.gain_matrix)

# %% MAKE A PLOT

# Change some plot settings (optional)
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{cmbright,amsmath}")
plt.rc("savefig", format="pdf")
plt.rc("savefig", bbox="tight")

# Plot the states (x) and input (u) vs time (t)
fig1 = plt.figure(1)
ax1a = plt.subplot(311)
plt.plot(t, x[0, :], "C0")
plt.step(t, x_hat[0, :], "C1--", where="post")
plt.grid(color="0.95")
plt.ylabel(r"$x_1$ [m]")
plt.setp(ax1a, xticklabels=[])
plt.legend(["Actual", "Estimated"])
ax1b = plt.subplot(312)
plt.plot(t, x[1, :], "C0")
plt.step(t, x_hat[1, :], "C1--", where="post")
plt.grid(color="0.95")
plt.ylabel(r"$x_2$ [m/s]")
plt.setp(ax1b, xticklabels=[])
ax1c = plt.subplot(313)
plt.step(t, u, "C1", where="post")
plt.grid(color="0.95")
plt.ylabel(r"$u$ [N]")
plt.xlabel(r"$t$ [s]")

# Save the plot
plt.savefig("../agv-book/figs/ch2/oneD_combined_control_fig1.pdf")

# %% MAKE AN ANIMATION

# Set the side length of the vehicle [m]
LENGTH = 1.0

# Let's use the Cart class to create an animation
vehicle = Cart(LENGTH)

# Create and save the animation
ani = vehicle.animate(
    x[0, :], T, True, "../agv-book/gifs/ch2/oneD_combined_control.gif"
)

# %%

# Show all the plots to the screen
plt.show()
