"""
Example vanilla_SLAM.py
Author: Joshua A. Marshall <joshua.marshall@queensu.ca>
GitHub: https://github.com/botprof/agv-examples
"""

# %%
# SIMULATION SETUP

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from matplotlib import patches

# Set the simulation time [s] and the sample period [s]
SIM_TIME = 60.0
T = 0.01

# Create an array of time values [s]
t = np.arange(0.0, SIM_TIME, T)
N = np.size(t)

# %%
# CREATE A MAP OF FEATURES TO BE MAPPED AND USED FOR LOCALIZATION

# Set the map expanse (square) [m]
d_map = 20

# Set the number of features
m = 10

# Create the map of features
f_map = np.zeros((2, m))
for i in range(1, m):
    f_map[:, i] = d_map * np.random.rand(2)

# %%
# VEHICLE, SENSOR MODELS AND KALMAN FILTER FUNCTIONS

# Discrete-time omnidirectional vehicle model
def vehicle(x, u, T):
    x_new = x + T * u
    return x_new


# Function to measure relative position to features in map
def f_sensor(x, f_map, R):

    # Find the number of features available
    m = np.shape(f_map)[1]

    # Find the relative position measurement with measurement noise
    y = np.zeros(2 * m)
    for i in range(0, m):
        y[2 * i] = f_map[0, i] - x[0] + np.sqrt(R[0, 0]) * np.random.randn()
        y[2 * i + 1] = f_map[1, i] - x[1] + np.sqrt(R[1, 1]) * np.random.randn()

    # Return the measurement array
    return y


# Prediction step for a KF with process noise covariance Q
def KF_predict(x, u, F, Q, P, T):
    x_hat = vehicle(x, u + np.sqrt(Q) @ np.random.randn(2), T)
    P_hat = F @ P @ np.transpose(F) + Q
    P_hat = (P_hat + np.transpose(P_hat)) / 2
    return x_hat, P_hat


# Correction step for a KF with measurement noise covariance R
def KF_correct(x, y, H, R, P):
    K = P @ np.transpose(H) @ np.linalg.inv(H @ P @ np.transpose(H) + R)
    x_hat = x + K @ (y - H @ x)
    P_hat = (np.identity(np.shape(x)[0]) - K @ H) @ P
    P_hat = (P_hat + np.transpose(P_hat)) / 2
    return x_hat, P_hat


# %%
# SET UP AND RUN THE SLAM PROBLEM

# Initial robot position [m, m]
x_init = np.zeros(2)

# Initialize the states and covariance arrays
x_veh = np.zeros((2, N))
x_hat = np.zeros((2 * m, N))
P_hat = np.zeros((2 * m, 2 * m, N))

# Set the process and measurement noise covariances
Q_x = np.diag(np.square([0.01, 0.01]))
R_y = np.diag(np.square([5.0, 5.0]))

# Choose a location where the vehicle starts (relative to map origin)
x_veh[:, 0] = -f_map[:, 0] + np.array([5, 5])

# Set up the vehicle model
F = np.eye(2)

# Set up the measurement model
H = np.zeros((2 * m, 2 * m))
H[0:2, :] = np.hstack((-np.identity(2), np.zeros((2, 2 * m - 2))))
H[2 : 2 * m, :] = np.hstack(
    (np.kron(np.ones((m - 1, 1)), -np.identity(2)), np.identity(2 * m - 2))
)
R = np.kron(np.identity(m), R_y)

# Simulation the SLAM solution
for i in range(0, N):

    # Initialize using the first feature as a point of reference
    if i == 0:

        # Feature observations made by the vehicle
        y = f_sensor(x_veh[:, i], f_map, R_y)

        # Initial state estimates
        x_hat[0:2, i] = -y[0:2]
        P_hat[0:2, 0:2, i] = Q_x + R_y
        for j in range(1, m):
            x_hat[2 * j, i] = x_hat[0, i] + y[2 * j]
            x_hat[2 * j + 1, i] = x_hat[1, i] + y[2 * j + 1]
            P_hat[2 * j : 2 * j + 2, 2 * j : 2 * j + 2, i] = P_hat[0:2, 0:2, i] + R_y

    else:

        # Compute some inputs (i.e., drive the vehicle around a square)
        if i < 0.25 * N:
            u = np.array([1, 0])
        elif i < 0.5 * N:
            u = np.array([0, 1])
        elif i < 0.75 * N:
            u = np.array([-1, 0])
        else:
            u = np.array([0, -1])

        # Update the vehicle motion
        x_veh[:, i] = vehicle(x_veh[:, i - 1], u, T)

        # Feature measurements/observations made by the vehicle
        y = f_sensor(x_veh[:, i], f_map, R_y)

        # Run the KF prediction step
        x_hat[0:2, i], P_hat[0:2, 0:2, i] = KF_predict(
            x_hat[0:2, i - 1], u, F, Q_x, P_hat[0:2, 0:2, i - 1], T
        )
        x_hat[2 : 2 * m, i] = x_hat[2 : 2 * m, i - 1]
        P_hat[0 : 2 * m, 0 : 2 * m, i] = P_hat[0 : 2 * m, 0 : 2 * m, i - 1]

        # Run the KF correction step
        x_hat[:, i], P_hat[:, :, i] = KF_correct(x_hat[:, i], y, H, R, P_hat[:, :, i])

# %%
# PLOT THE SIMULATION OUTPUTS

# Find the scaling factor for plotting covariance bounds
alpha = 0.05
s1 = chi2.isf(alpha, 1)
s2 = chi2.isf(alpha, 2)

# Plot the errors and covariance bounds for the vehicle state
sigma = np.zeros((2, N))
fig1 = plt.figure(1)
ax1 = plt.subplot(211)
plt.plot(t, x_veh[0, :] - x_hat[0, :], "C0", label="Error")
sigma[0, :] = np.sqrt(s1 * P_hat[0, 0, :])
plt.fill_between(
    t,
    -sigma[0, :],
    sigma[0, :],
    color="C1",
    alpha=0.2,
    label=str(100 * (1 - alpha)) + "% confidence",
)
plt.ylabel(r"$e_1$ [m]")
plt.setp(ax1, xticklabels=[])
ax1.set_ylim([-2.5, 2.5])
plt.legend()
plt.grid(color="0.95")
ax2 = plt.subplot(212)
plt.plot(t, x_veh[1, :] - x_hat[1, :], "C0")
sigma[1, :] = np.sqrt(s1 * P_hat[1, 1, :])
plt.fill_between(t, -sigma[1, :], sigma[1, :], color="C1", alpha=0.2)
plt.ylabel(r"$e_2$ [m]")
plt.xlabel(r"$t$ [s]")
ax2.set_ylim([-2.5, 2.5])
plt.grid(color="0.95")

# Plot the actual and estimate vehicle poses
fig2, ax = plt.subplots()
plt.plot(x_veh[0, :], x_veh[1, :], "C0", label="Actual")
plt.plot(x_hat[0, :], x_hat[1, :], "C1", label="Estimated")

# Plot each feature's estimated location and covariance
for i in range(1, m):
    plt.plot(x_hat[2 * i, N - 1], x_hat[2 * i + 1, N - 1], "C1*")
    W, V = np.linalg.eig(P_hat[2 * i : 2 * i + 2, 2 * i : 2 * i + 2, N - 1])
    j_max = np.argmax(W)
    j_min = np.argmin(W)
    ell = patches.Ellipse(
        (x_hat[2 * i, N - 1], x_hat[2 * i + 1, N - 1]),
        2 * np.sqrt(s2 * W[j_max]),
        2 * np.sqrt(s2 * W[j_min]),
        angle=np.arctan2(V[j_max, 1], V[j_max, 0]),
        alpha=0.1,
        color="C1",
    )
    ax.add_artist(ell)

# Plot the actual feature locations
plt.plot(f_map[0, :], f_map[1, :], "C0*", label="Map feature")
plt.xlabel(r"$x_1$ [m]")
plt.ylabel(r"$x_2$ [m]")
plt.legend()
plt.grid(color="0.95")
plt.axis("equal")

# Show the plots to the screen
plt.show()
