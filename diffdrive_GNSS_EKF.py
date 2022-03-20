"""
Example diffdrive_GNSS_EKF.py
Author: Joshua A. Marshall <joshua.marshall@queensu.ca>
GitHub: https://github.com/botprof/agv-examples
"""

# %%
# SIMULATION SETUP

import numpy as np
import matplotlib.pyplot as plt
from mobotpy.models import DiffDrive
from mobotpy.integration import rk_four
from scipy.stats import chi2
from numpy.linalg import inv

# Set the simulation time [s] and the sample period [s]
SIM_TIME = 30.0
T = 0.01

# Create an array of time values [s]
t = np.arange(0.0, SIM_TIME, T)
N = np.size(t)

# %%
# VEHICLE SETUP

# Set the track length of the vehicle [m]
ELL = 1.0

# Create a vehicle object of type DiffDrive
vehicle = DiffDrive(ELL)

# %%
# FUNCTION DEFINITIONS


def unicycle_GNSS_ekf(u_m, y, Q, R, x, P, T):

    # Define some matrices for the a priori step
    G = np.array([[np.cos(x[2]), 0], [np.sin(x[2]), 0], [0, 1]])
    F = np.identity(3) + T * np.array(
        [[0, 0, -u_m[0] * np.sin(x[2])], [0, 0, u_m[0] * np.cos(x[2])], [0, 0, 0]]
    )
    L = T * G

    # Compute a priori estimate
    x_new = x + T * G @ u_m
    P_new = F @ P @ np.transpose(F) + L @ Q @ np.transpose(L)

    # Numerically help the covariance matrix stay symmetric
    P_new = (P_new + np.transpose(P_new)) / 2

    # Define some matrices for the a posteriori step
    C = np.array([[1, 0, 0], [0, 1, 0]])
    H = C
    M = np.identity(2)

    # Compute the a posteriori estimate
    K = (
        P_new
        @ np.transpose(H)
        @ inv(H @ P_new @ np.transpose(H) + M @ R @ np.transpose(M))
    )
    x_new = x_new + K @ (y - C @ x_new)
    P_new = (np.identity(3) - K @ H) @ P_new

    # Numerically help the covariance matrix stay symmetric
    P_new = (P_new + np.transpose(P_new)) / 2

    # Define the function output
    return x_new, P_new


# %%
# SET UP INITIAL CONDITIONS AND ESTIMATOR PARAMETERS

# Initial conditions
x_init = np.zeros(3)
x_init[0] = 0.0
x_init[1] = 0.0
x_init[2] = 0.0

# Setup some arrays for the actual vehicle
x = np.zeros((3, N))
u = np.zeros((2, N))
x[:, 0] = x_init

# Set the initial guess for the estimator
x_guess = x_init + np.array([5.0, -5.0, 0.1])

# Set the initial pose covariance estimate as a diagonal matrix
P_guess = np.diag(np.square([5.0, -5.0, 0.1]))

# Set the true process and measurement noise covariances
Q = np.diag(
    [
        1.0 / (2.0 * np.power(T, 2)) * np.power(0.2 * np.pi / ((2 ** 12) * 3), 2),
        np.power(0.001, 2),
    ]
)
R = np.power(5.0, 2) * np.identity(2)

# Initialized estimator arrays
x_hat = np.zeros((3, N))
x_hat[:, 0] = x_guess

# Measured odometry (speed and angular rate) and GNSS (x, y) signals
u_m = np.zeros((2, N))
y = np.zeros((2, N))

# Covariance of the estimate
P_hat = np.zeros((3, 3, N))
P_hat[:, :, 0] = P_guess

# Compute some inputs to just drive around
for k in range(1, N):
    # Compute some inputs to steer the unicycle around
    u_unicycle = np.array([2.0, np.sin(0.005 * T * k)])

# %%
# SIMULATE AND PLOT WITHOUT GNSS

# Set the process and measurement noise covariances to ignore GNSS
Q_hat = Q
R_hat = 1e10 * R

for k in range(1, N):

    # Simulate the differential drive vehicle's motion
    u[:, k] = vehicle.uni2diff(u_unicycle)
    x[:, k] = rk_four(vehicle.f, x[:, k - 1], u[:, k - 1], T)

    # Simulate the vehicle speed estimate
    u_m[0, k] = u_unicycle[0] + np.power(Q[0, 0], 0.5) * np.random.randn(1)

    # Simulate the angular rate gyroscope measurement
    u_m[1, k] = u_unicycle[1] + np.power(Q[1, 1], 0.5) * np.random.randn(1)

    # Simulate the GNSS measurement
    y[0, k] = x[0, k] + np.power(R[0, 0], 0.5) * np.random.randn(1)
    y[1, k] = x[1, k] + np.power(R[1, 1], 0.5) * np.random.randn(1)

    # Run the EKF estimator
    x_hat[:, k], P_hat[:, :, k] = unicycle_GNSS_ekf(
        u_m[:, k], y[:, k], Q_hat, R_hat, x_hat[:, k - 1], P_hat[:, :, k - 1], T
    )

# Change some plot settings (optional)
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{cmbright,amsmath,bm}")
plt.rc("savefig", format="pdf")
plt.rc("savefig", bbox="tight")

# Plot results without GNSS
fig1 = plt.figure(1)
ax1 = plt.subplot(311)
plt.setp(ax1, xticklabels=[])
plt.plot(t, x[0, :], "C0", label="Actual")
plt.plot(t, x_hat[0, :], "C1--", label="Estimated")
plt.ylabel(r"$x_1$ [m]")
plt.grid(color="0.95")
plt.legend()
ax2 = plt.subplot(312)
plt.setp(ax2, xticklabels=[])
plt.plot(t, x[1, :], "C0")
plt.plot(t, x_hat[1, :], "C1--")
plt.ylabel(r"$x_2$ [m]")
plt.grid(color="0.95")
ax3 = plt.subplot(313)
plt.plot(t, x[2, :], "C0")
plt.plot(t, x_hat[2, :], "C1--")
plt.xlabel(r"$t$ [s]")
plt.ylabel(r"$x_3$ [rad]")
plt.grid(color="0.95")

# Save the plot
plt.savefig("../agv-book/figs/ch5/diffdrive_GNSS_EKF_fig1.pdf")

# Find the scaling factor for plotting 99% covariance bounds
alpha = 0.01
s1 = chi2.isf(alpha, 1)
s2 = chi2.isf(alpha, 2)

# Plot the estimation error with covariance bounds
sigma = np.zeros((3, N))
fig2 = plt.figure(2)
ax1 = plt.subplot(311)
sigma[0, :] = np.sqrt(s1 * P_hat[0, 0, :])
plt.fill_between(
    t,
    -sigma[0, :],
    sigma[0, :],
    color="C0",
    alpha=0.2,
    label=str(100 * (1 - alpha)) + " \% Confidence",
)
plt.plot(t, x[0, :] - x_hat[0, :], "C0", label="Error")
plt.ylabel(r"$e_1$ [m]")
plt.setp(ax1, xticklabels=[])
plt.grid(color="0.95")
plt.legend()
ax2 = plt.subplot(312)
sigma[1, :] = np.sqrt(s1 * P_hat[1, 1, :])
plt.fill_between(
    t,
    -sigma[1, :],
    sigma[1, :],
    color="C0",
    alpha=0.2,
    label=str(100 * (1 - alpha)) + " \% Confidence",
)
plt.plot(t, x[1, :] - x_hat[1, :], "C0", label="Error")
plt.ylabel(r"$e_2$ [m]")
plt.setp(ax2, xticklabels=[])
plt.grid(color="0.95")
ax3 = plt.subplot(313)
sigma[2, :] = np.sqrt(s1 * P_hat[2, 2, :])
plt.fill_between(
    t,
    -sigma[2, :],
    sigma[2, :],
    color="C0",
    alpha=0.2,
    label=str(100 * (1 - alpha)) + " \% Confidence",
)
plt.plot(t, x[2, :] - x_hat[2, :], "C0", label="Error")
plt.ylabel(r"$e_3$ [rad]")
plt.xlabel(r"$t$ [s]")
plt.grid(color="0.95")

# Save the plot
plt.savefig("../agv-book/figs/ch5/diffdrive_GNSS_EKF_fig2.pdf")

# Show the plots to the screen
plt.show()

# %%
# PLOT THE NOISY GNSS DATA

fig3 = plt.figure(3)
ax1 = plt.subplot(211)
plt.plot(t, y[0, :], "C1", label="GNSS measurement")
plt.plot(t, x[0, :], "C0", label="Actual")
plt.ylabel(r"$x_1$ [m]")
plt.grid(color="0.95")
plt.setp(ax1, xticklabels=[])
plt.legend()
ax2 = plt.subplot(212)
plt.plot(t, y[1, :], "C1", label="GNSS measurement")
plt.plot(t, x[1, :], "C0", label="Actual")
plt.ylabel(r"$x_2$ [m]")
plt.xlabel(r"$t$ [s]")
plt.grid(color="0.95")

# Save the plot
plt.savefig("../agv-book/figs/ch5/diffdrive_GNSS_EKF_fig3.pdf")

# Show the plots to the screen
plt.show()

# %%
# SIMULATE AND PLOT WITH GNSS + ODOMETRY FUSION

# Find the scaling factor for plotting covariance bounds
alpha = 0.01
s1 = chi2.isf(alpha, 1)
s2 = chi2.isf(alpha, 2)

# Estimate the process and measurement noise covariances
Q_hat = Q
R_hat = R

for k in range(1, N):

    # Simulate the differential drive vehicle's motion
    u[:, k] = vehicle.uni2diff(u_unicycle)
    x[:, k] = rk_four(vehicle.f, x[:, k - 1], u[:, k - 1], T)

    # Simulate the vehicle speed estimate
    u_m[0, k] = u_unicycle[0] + np.power(Q[0, 0], 0.5) * np.random.randn(1)

    # Simulate the angular rate gyroscope measurement
    u_m[1, k] = u_unicycle[1] + np.power(Q[1, 1], 0.5) * np.random.randn(1)

    # Simulate the GNSS measurement
    y[0, k] = x[0, k] + np.power(R[0, 0], 0.5) * np.random.randn(1)
    y[1, k] = x[1, k] + np.power(R[1, 1], 0.5) * np.random.randn(1)

    # Run the EKF estimator
    x_hat[:, k], P_hat[:, :, k] = unicycle_GNSS_ekf(
        u_m[:, k], y[:, k], Q_hat, R_hat, x_hat[:, k - 1], P_hat[:, :, k - 1], T
    )

# Set the ranges for error uncertainty axes
x_range = 2.0
y_range = 2.0
theta_range = 0.2

# Plot the estimation error with covariance bounds
sigma = np.zeros((3, N))
fig4 = plt.figure(4)
ax1 = plt.subplot(311)
sigma[0, :] = np.sqrt(s1 * P_hat[0, 0, :])
plt.fill_between(
    t,
    -sigma[0, :],
    sigma[0, :],
    color="C0",
    alpha=0.2,
    label=str(100 * (1 - alpha)) + " \% Confidence",
)
plt.plot(t, x[0, :] - x_hat[0, :], "C0", label="Error")
plt.ylabel(r"$e_1$ [m]")
plt.setp(ax1, xticklabels=[])
ax1.set_ylim([-x_range, x_range])
plt.grid(color="0.95")
plt.legend()
ax2 = plt.subplot(312)
sigma[1, :] = np.sqrt(s1 * P_hat[1, 1, :])
plt.fill_between(t, -sigma[1, :], sigma[1, :], color="C0", alpha=0.2)
plt.plot(t, x[1, :] - x_hat[1, :], "C0")
plt.ylabel(r"$e_2$ [m]")
plt.setp(ax2, xticklabels=[])
ax2.set_ylim([-y_range, y_range])
plt.grid(color="0.95")
ax3 = plt.subplot(313)
sigma[2, :] = np.sqrt(s1 * P_hat[2, 2, :])
plt.fill_between(
    t,
    -sigma[2, :],
    sigma[2, :],
    color="C0",
    alpha=0.2,
    label=str(100 * (1 - alpha)) + " \% Confidence",
)
plt.plot(t, x[2, :] - x_hat[2, :], "C0", label="Error")
ax3.set_ylim([-theta_range, theta_range])
plt.ylabel(r"$e_3$ [rad]")
plt.xlabel(r"$t$ [s]")
plt.grid(color="0.95")

# Save the plot
plt.savefig("../agv-book/figs/ch5/diffdrive_GNSS_EKF_fig4.pdf")

# Show the plots to the screen
plt.show()

# %%
# MAKE AN ANIMATION

# Create and save the animation
ani = vehicle.animate_estimation(
    x, x_hat, P_hat, alpha, T, True, "../agv-book/gifs/ch5/diffdrive_GNSS_EKF.gif"
)

# Show the movie to the screen
plt.show()

# # Show animation in HTML output if you are using IPython or Jupyter notebooks
# plt.rc('animation', html='jshtml')
# display(ani)
# plt.close()
