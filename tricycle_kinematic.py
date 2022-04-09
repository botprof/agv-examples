"""
Example tricycle_kinematic.py
Author: Joshua A. Marshall <joshua.marshall@queensu.ca>
GitHub: https://github.com/botprof/agv-examples
"""

# %%
# SIMULATION SETUP

import numpy as np
import matplotlib.pyplot as plt
from mobotpy.models import Tricycle
from mobotpy.integration import rk_four

# Set the simulation time [s] and the sample period [s]
SIM_TIME = 15.0
T = 0.1

# Create an array of time values [s]
t = np.arange(0.0, SIM_TIME, T)
N = np.size(t)

# Set the wheelbase and track of the vehicle [m]
ELL_W = 2.50
ELL_T = 1.75

# %%
# MODEL DEFINTION


def tricycle_f(x, u):
    """Tricycle vehicle kinematic model."""
    f = np.zeros(4)
    f[0] = u[0] * np.cos(x[2])
    f[1] = u[0] * np.sin(x[2])
    f[2] = u[0] * 1.0 / ELL_W * np.tan(x[3])
    f[3] = u[1]
    return f


# %%
# RUN SIMULATION

# Initialize arrays that will be populated with our inputs and states
x = np.zeros((4, N))
u = np.zeros((2, N))

# Set the initial pose [m, m, rad, rad], velocities [m/s, rad/s]
x[0, 0] = 0.0
x[1, 0] = 0.0
x[2, 0] = np.pi / 2.0
x[3, 0] = 0.0
u[0, 0] = 5.0
u[1, 0] = 0

# Run the simulation
for k in range(1, N):
    x[:, k] = rk_four(tricycle_f, x[:, k - 1], u[:, k - 1], T)
    u[0, k] = 5.0
    u[1, k] = 0.25 * np.sin(2.0 * t[k])

# %%
# MAKE SOME PLOTS

# Change some plot settings (optional)
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{cmbright,amsmath,bm}")
plt.rc("savefig", format="pdf")
plt.rc("savefig", bbox="tight")

# Plot the states as a function of time
fig1 = plt.figure(1)
fig1.set_figheight(6.4)
ax1a = plt.subplot(611)
plt.plot(t, x[0, :])
plt.grid(color="0.95")
plt.ylabel(r"$x$ [m]")
plt.setp(ax1a, xticklabels=[])
ax1b = plt.subplot(612)
plt.plot(t, x[1, :])
plt.grid(color="0.95")
plt.ylabel(r"$y$ [m]")
plt.setp(ax1b, xticklabels=[])
ax1c = plt.subplot(613)
plt.plot(t, x[2, :] * 180.0 / np.pi)
plt.grid(color="0.95")
plt.ylabel(r"$\theta$ [deg]")
plt.setp(ax1c, xticklabels=[])
ax1c = plt.subplot(614)
plt.plot(t, x[3, :] * 180.0 / np.pi)
plt.grid(color="0.95")
plt.ylabel(r"$\phi$ [deg]")
plt.setp(ax1c, xticklabels=[])
ax1c = plt.subplot(615)
plt.step(t, u[0, :], "C1", where="post")
plt.grid(color="0.95")
plt.ylabel(r"$v_1$ [m/s]")
plt.setp(ax1c, xticklabels=[])
ax1d = plt.subplot(616)
plt.step(t, u[1, :], "C1", where="post")
plt.grid(color="0.95")
plt.ylabel(r"$v_2$ [deg/s]")
plt.xlabel(r"$t$ [s]")

# Save the plot
plt.savefig("../agv-book/figs/ch3/tricycle_kinematic_fig1.pdf")

# Let's now use the class Tricycle for plotting
vehicle = Tricycle(ELL_W, ELL_T)

# Plot the position of the vehicle in the plane
fig2 = plt.figure(2)
plt.plot(x[0, :], x[1, :])
plt.axis("equal")
X_L, Y_L, X_R, Y_R, X_F, Y_F, X_B, Y_B = vehicle.draw(
    x[0, 0], x[1, 0], x[2, 0], x[3, 0]
)
plt.fill(X_L, Y_L, "k")
plt.fill(X_R, Y_R, "k")
plt.fill(X_F, Y_F, "k")
plt.fill(X_B, Y_B, "C2", alpha=0.5, label="Start")
X_L, Y_L, X_R, Y_R, X_F, Y_F, X_B, Y_B = vehicle.draw(
    x[0, N - 1], x[1, N - 1], x[2, N - 1], x[3, N - 1]
)
plt.fill(X_L, Y_L, "k")
plt.fill(X_R, Y_R, "k")
plt.fill(X_F, Y_F, "k")
plt.fill(X_B, Y_B, "C3", alpha=0.5, label="End")
plt.xlabel(r"$x$ [m]")
plt.ylabel(r"$y$ [m]")
plt.legend()

# Save the plot
plt.savefig("../agv-book/figs/ch3/tricycle_kinematic_fig2.pdf")

# Show all the plots to the screen
plt.show()

# %%
# MAKE AN ANIMATION

# Create and save the animation
ani = vehicle.animate(x, T, True, "../agv-book/gifs/ch3/tricycle_kinematic.gif")

# Show the movie to the screen
plt.show()

# # Show animation in HTML output if you are using IPython or Jupyter notebooks
# plt.rc('animation', html='jshtml')
# display(ani)
# plt.close()
