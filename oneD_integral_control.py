"""
Example oneD_integral_control.py
Author: Joshua A. Marshall <joshua.marshall@queensu.ca>
GitHub: https://github.com/botprof/agv-examples
"""

# %% 
# SIMULATION SETUP

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from mobotpy.models import Cart

# %% 
# PARAMETERS

# Set some variables that describe the desired behaviour
ZETA = 1.1
OMEGA_N = np.sqrt(0.3)

# Define the sample time [s]
T = 0.1

# Compute the pole locations
lambda_s = np.roots([1, 2 * ZETA * OMEGA_N, OMEGA_N ** 2])
lambda_z = np.hstack((np.exp(lambda_s * T), 0.5))

# Define the vehicle mass [kg]
M = 10.0

# Define the system matrices
F = np.array([[1, T], [0, 1]])
G = np.array([[T ** 2 / (2 * M)], [T / M]])
H = np.array([[-1, 0]])

# Augmented state system matrices
A = np.hstack((np.vstack((F, H)), np.array([[0], [0], [1]])))
B = np.vstack((G, 0))

# Find gain matrix K that places the poles inside the unit disk
K = signal.place_poles(A, B, lambda_z)

# %% 
# FUNCTION DEFINITIONS


def vehicle(x, u, F, G):
    """Discrete-time 1D vehicle model on a slope."""
    x_new = F @ x + G @ [u] - G @ [M * 9.81 * np.sin(np.pi / 30)]
    return x_new


def integrator(x, xi):
    """Augmented state integrator."""
    xi_new = xi - x[0]
    return xi_new


def controller(x, xi, K):
    """State feedback controller with integral action."""
    u = -K @ np.hstack((x, xi))
    return u


# %% 
# RUN SIMULATION

# Create an array of time values [s]
SIM_TIME = 15.0
t = np.arange(0, SIM_TIME, T)
N = np.size(t)

# Initialize arrays that will be populated with our inputs and states
x = np.zeros((2, N))
xi = np.zeros(N)
u = np.zeros(N)

# Set the initial position [m], velocity [m/s], and force input [N]
x[0, 0] = 1.0
x[1, 0] = 0.0
u[0] = 0.0

# Run the simulation
for k in range(1, N):
    x[:, k] = vehicle(x[:, k - 1], u[k - 1], F, G)
    xi[k] = integrator(x[:, k - 1], xi[k - 1])
    u[k] = controller(x[:, k], xi[k], K.gain_matrix)

# %% 
# MAKE A PLOT

# Change some plot settings (optional)
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{cmbright,amsmath}")
plt.rc("savefig", format="pdf")
plt.rc("savefig", bbox="tight")

# Plot the states (x) and input (u) vs time (t)
fig1 = plt.figure(1)
ax1a = plt.subplot(311)
plt.plot(t, x[0, :], "C0")
plt.grid(color="0.95")
plt.ylabel(r"$x_1$ [m]")
plt.setp(ax1a, xticklabels=[])
ax1b = plt.subplot(312)
plt.plot(t, x[1, :], "C0")
plt.grid(color="0.95")
plt.ylabel(r"$x_2$ [m/s]")
plt.setp(ax1b, xticklabels=[])
ax1c = plt.subplot(313)
plt.step(t, u, "C1", where="post")
plt.grid(color="0.95")
plt.ylabel(r"$u$ [N]")
plt.xlabel(r"$t$ [s]")

# Save the plot
plt.savefig("../agv-book/figs/ch2/oneD_integral_control_fig1.pdf")

# %% 
# MAKE AN ANIMATION

# Set the side length of the vehicle [m]
LENGTH = 1.0

# Let's use the Cart class to create an animation
vehicle = Cart(LENGTH)

# Create and save the animation
ani = vehicle.animate(
    x[0, :], T, True, "../agv-book/gifs/ch2/oneD_integral_control.gif"
)

# %%

# Show all the plots to the screen
plt.show()
