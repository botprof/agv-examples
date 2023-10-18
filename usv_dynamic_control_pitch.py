"""
Example usv_dynamic_pitch_control.py
Author: Thomas M. C. Sears <thomas.sears@queensu.ca>
GitHub: https://github.com/botprof/agv-examples
"""

# %%
# SIMULATION SETUP

import numpy as np
import matplotlib.pyplot as plt
from mobotpy.models import LongitudinalUSV

# Set the simulation time [s] and the sample period [s]
SIM_TIME = 5.0
T = 0.02

# Create an array of time values [s]
t = np.arange(0, SIM_TIME, T)
N = np.size(t)


# %%
# FUNCTION DEFINITIONS
# Set the mass of the vehicle [kg]
m = 50.0

# Define the vehicle model matrices
F = np.array([[1, T], [0, 1]])
G = np.array([[T ** 2 / (2 * m)], [T / m]])


def vehicle(x, u, F, G):
    """Discrete-time 1D dynamic vehicle model."""
    x_new = F @ x + G @ [u]
    return x_new


def controller(x, K):
    """Proportional controller."""
    u = -K @ x
    return u


def wave(x, t):
    # Define the parameters of the sinusoidal array
    amplitude = 1.0  # [m]
    phase = np.pi / 2  # [rad]

    period = 1  # [s]
    time_frequency = 1 / period  # [1/s]
    wavelength = 10  # [m]
    space_frequency = 1 / wavelength  # [1/m]

    # Find height of wave
    height = amplitude * \
        np.sin(2*np.pi*(time_frequency*t + space_frequency*x) + phase)

    # Find slope of wave in space
    slope_x = 2*np.pi*space_frequency*amplitude * \
        np.cos(2*np.pi*(time_frequency*t + space_frequency*x) + phase)
    wave_angle = np.arctan2(slope_x, 1)

    return height, wave_angle

# %%
# RUN SIMULATION


# Initialize arrays that will be populated with our inputs and states
x = np.zeros((2, N))
u = np.zeros(N)

# Choose the controller gains
K = np.array([3.0, 4.0])

# Set the initial position [m], velocity [m/s], and force input [N]
x[0, 0] = 1.0
x[1, 0] = 0.0
u[0] = 0.0

# Run the simulation
for k in range(1, N):
    x[:, k] = vehicle(x[:, k - 1], u[k - 1], F, G)
    u[k] = controller(x[:, k], K)

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

plt.show()

# Save the plot
# plt.savefig("../agv-book/figs/ch2/oneD_dynamic_control_fig1.pdf")

# %%
# # TEMPORARY ADDITION OF PITCH FOR ANIMATION

x = np.zeros((2, N))
x[1, :], sin_array = wave(x[0, :], t)
new_x = np.vstack((x, sin_array))

# %%
# GET SPACE-TIME WAVE DATA

x_range = np.arange(-20, 20, 0.1)
N_x = np.size(x_range)

wave_heights = np.zeros((N_x, N))

for i in range(N):
    wave_heights[:, i], _ = wave(x_range, t[i])


# %%
# MAKE AN ANIMATION

# Set the side length of the vehicle [m]
LENGTH = 2.0

# Let's use the Cart class to create an animation
vehicle = LongitudinalUSV(LENGTH)

# Create and save the animation
ani = vehicle.animate(new_x, T,
                      wave_positions=x_range, wave_data=wave_heights,
                      relative=False)

# %%

# Show all the plots to the screen
plt.show()

# Show animation in HTML output if you are using IPython or Jupyter notebooks
plt.rc('animation', html='jshtml')
# plt.rc('animation')
display(ani)
plt.close()

# %%