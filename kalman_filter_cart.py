"""
Example kalman_filter_cart.py
Author: Joshua A. Marshall <joshua.marshall@queensu.ca>
GitHub: https://github.com/botprof/agv-examples
"""

# %%
# SIMULATION SETUP

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

# Set the simulation time [s] and the sample period [s]
SIM_TIME = 10.0
T = 0.1

# Create an array of time values [s]
t = np.arange(0.0, SIM_TIME, T)
N = np.size(t)

# %%
# DEFINE THE ROBOT AND SENSOR MODELS

# Mobile robot model
M = 1.0
F = np.array([[1, T], [0, 1]])
G = np.array([0.5 * T**2 / M, T / M])

# Sensor model (GPS-like sensor)
H = np.array([1, 0])

# Model noise covariance
Q = np.diag([0.1**2, 0.01**2])

# GPS sensor noise covariance
R = 2.5**2

# %%
# SET UP INITIAL CONDITIONS

# Set up arrays for a two-dimensional state
x_hat = np.zeros((2, N))
P_hat = np.zeros((2, 2, N))

# Robot state (actual)
x = np.zeros([2, N])
x[:, 0] = np.array([3.0, 0.0])

# Set an array for the sensor (GPS) measurements
y = np.zeros(N)

# Initial estimate of the robot's location
x_hat[:, 0] = x[:, 0] + np.array([2.0, 0.0])
# Initial covariance matrix
P_hat[:, :, 0] = np.diag([2.0**2, 0.2**2])

# Initial inputs
u = np.zeros([N])

# %%
# RUN THE SIMULATION

for k in range(1, N):
    # Simulate the actual vehicle
    x[:, k] = F @ x[:, k - 1] + G * u[k - 1]

    # Run the a priori estimation step
    x_hat[:, k] = F @ x_hat[:, k - 1] + G * (u[k - 1] + np.sqrt(Q) @ np.random.standard_normal(2))
    P_hat[:, :, k] = F @ P_hat[:, :, k - 1] @ F.T + Q
    # Help the covariance matrix stay symmetric
    P_hat[:, :, k] = 0.5 * (P_hat[:, :, k] + P_hat[:, :, k].T)

    # Take a sensor measurement (with zero-mean Gaussian noise)
    y[k] = H @ x[:, k] + np.sqrt(R) * np.random.standard_normal()

    # Run the a posteriori estimation step
    K = P_hat[:, :, k] @ H.T / (H @ P_hat[:, :, k] @ H.T + R)
    x_hat[:, k] = x_hat[:, k] + K * (y[k] - H @ x_hat[:, k])

    P_hat[:, :, k] = (np.eye(2) - K @ H) @ P_hat[:, :, k] @ (
        np.eye(2) - K @ H
    ).T + K @ K.T * R
    # Help the covariance matrix stay symmetric
    P_hat[:, :, k] = 0.5 * (P_hat[:, :, k] + P_hat[:, :, k].T)

    # Control input (actual)
    u[k] = 10 * np.sin(2 * t[k])

# %%
# PLOT THE RESULTS

# Change some plot settings (optional)
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{cmbright,amsmath,bm}")
plt.rc("savefig", format="pdf")
plt.rc("savefig", bbox="tight")

# Find the scaling factor for plotting covariance bounds
ALPHA = 0.01
s1 = chi2.isf(ALPHA, 1)
s2 = chi2.isf(ALPHA, 2)

# Plot the true position and noisy sensor (GPS) measurements
plt.figure()
plt.plot(t, y, "C2", label="Sensor")
plt.plot(t, x[0, :], "C0", label="Actual")
plt.xlabel(r"$t$ [s]")
plt.ylabel(r"$x_1$ [m]")
plt.grid(color="0.95")
plt.legend()

# Plot the state and the estimate
sigma = np.zeros((2, N))
plt.figure()
ax1 = plt.subplot(211)
sigma[0, :] = np.sqrt(s1 * P_hat[0, 0, :])
plt.fill_between(
    t,
    x_hat[0, :] - sigma[0, :],
    x_hat[0, :] + sigma[0, :],
    color="C0",
    alpha=0.2,
    label=str(100 * (1 - ALPHA)) + r" \% Confidence",
)
plt.plot(t, x[0, :], "C0", label="Actual")
plt.plot(t, x_hat[0, :], "C1", label="Estimate")
plt.plot(t, y, "C2", label="Sensor", alpha=0.2)
plt.ylabel(r"$x_1$ [m]")
plt.setp(ax1, xticklabels=[])
plt.grid(color="0.95")
plt.legend()
ax2 = plt.subplot(212)
sigma[1, :] = np.sqrt(s1 * P_hat[1, 1, :])
plt.fill_between(
    t,
    x_hat[1, :] - sigma[1, :],
    x_hat[1, :] + sigma[1, :],
    color="C0",
    alpha=0.2,
    label=str(100 * (1 - ALPHA)) + r" \% Confidence",
)
plt.plot(t, x[1, :], "C0", label="Actual")
plt.plot(t, x_hat[1, :], "C1", label="Estimate")
plt.ylabel(r"$x_2$ [m]")
plt.grid(color="0.95")
plt.xlabel(r"$t$ [s]")
plt.show()

# Plot the error in the estimate
sigma = np.zeros((2, N))
plt.figure()
ax1 = plt.subplot(211)
sigma[0, :] = np.sqrt(s1 * P_hat[0, 0, :])
plt.fill_between(
    t,
    -sigma[0, :],
    sigma[0, :],
    color="C0",
    alpha=0.2,
    label=str(100 * (1 - ALPHA)) + r" \% Confidence",
)
plt.plot(t, x[0, :] - x_hat[0, :], "C0", label="Estimator error")
plt.plot(t, x[0, :] - y, "C2", label="GPS-like sensor error", alpha=0.2)
plt.ylabel(r"$e_1$ [m]")
plt.setp(ax1, xticklabels=[])
plt.grid(color="0.95")
plt.legend()
ax2 = plt.subplot(212)
sigma[1, :] = np.sqrt(s1 * P_hat[1, 1, :])
plt.fill_between(
    t,
    -sigma[1, :],
    sigma[1, :],
    color="C0",
    alpha=0.2,
    label=str(100 * (1 - ALPHA)) + r" \% Confidence",
)
plt.plot(t, x[1, :] - x_hat[1, :], "C0", label="Estimator Error")
plt.ylabel(r"$e_2$ [m]")
plt.grid(color="0.95")
plt.xlabel(r"$t$ [s]")
plt.show()
