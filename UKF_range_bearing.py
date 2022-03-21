"""
Example UKF_range_bearing.py
Author: Joshua A. Marshall <joshua.marshall@queensu.ca>
GitHub: https://github.com/botprof/agv-examples
"""

# %%
# SIMULATION SETUP

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.stats import chi2
from numpy.linalg import inv
from scipy.linalg import block_diag
from mobotpy.integration import rk_four
from mobotpy.models import DiffDrive

# Set the simulation time [s] and the sample period [s]
SIM_TIME = 30.0
T = 0.04

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
# BUILD A MAP OF FEATURES IN THE VEHICLE'S ENVIRONMENT

# Number of features
M = 150

# Map size [m]
D_MAP = 100

# Randomly place features in the map
f_map = np.zeros((2, M))
for i in range(0, M):
    f_map[:, i] = D_MAP * (2.0 * np.random.rand(2) - 1.0)

# %%
# SENSOR MODELS

# Uncertainty in wheel speeds [m/s]
SIGMA_SPEED = 0.1

# Max and min sensor ranges
R_MAX = 15
R_MIN = 1

# Set the range [m] and bearing [rad] uncertainty
SIGMA_RANGE = 0.3
SIGMA_BEARING = 15 * np.pi / 180

# Create a range and bearing sensor model
def RandB_sensor(x, f_map, R):

    # Define how many total features are available
    m = np.shape(f_map)[1]

    # Find the indices of features that are within range [r_min, r_max]
    a = np.array([])
    for i in range(0, m):
        r = np.sqrt((f_map[0, i] - x[0]) ** 2 + (f_map[1, i] - x[1]) ** 2)
        if np.all(
            [
                r < R_MAX,
                r > R_MIN,
            ]
        ):
            a = np.append(a, i)

    # Compute the range and bearing to all features within range
    if np.shape(a)[0] > 0:
        # Specify the size of the output
        m_k = np.shape(a)[0]
        y = np.zeros(2 * m_k)

        # Compute the range and bearing to all features (including sensor noise)
        for i in range(0, m_k):
            # Range measurement [m]
            y[2 * i] = np.sqrt(
                (f_map[0, int(a[i])] - x[0]) ** 2 + (f_map[1, int(a[i])] - x[1]) ** 2
            ) + np.sqrt(R[0, 0]) * np.random.randn(1)
            # Bearing measurement [rad]
            y[2 * i + 1] = (
                np.unwrap(
                    np.array(
                        [
                            np.arctan2(
                                f_map[1, int(a[i])] - x[1], f_map[0, int(a[i])] - x[0]
                            )
                            - x[2]
                        ]
                    )
                )
                - np.pi
                + np.sqrt(R[1, 1]) * np.random.randn(1)
            )
    else:
        # No features were found within the sensing range
        y = np.array([])

    # Return the range and bearing to features in y with indices in a
    return y, a


# %%
# CREATE A UKF-BASED ESTIMATOR


def UKF(x, P, v_m, y_m, a, f_map, Q, R, kappa):

    # Set the augmented state and covariance
    xi = np.append(x, v_m)
    n_x = np.shape(x)[0]
    n_xi = np.shape(xi)[0]
    P_xi = block_diag(P, Q)

    # Define a set of sigma points for for the a priori estimate
    xi_sig = np.zeros((n_xi, 2 * n_xi + 1))
    P_xi_sig = np.linalg.cholesky((n_xi + kappa) * P_xi)
    for i in range(0, n_xi):
        xi_sig[:, i + 1] = xi + P_xi_sig[:, i]
        xi_sig[:, n_xi + i + 1] = xi - P_xi_sig[:, i]

    # Propagate each sigma point through the vehicle's model
    xi_sig_hat = np.zeros((n_xi, 2 * n_xi + 1))
    for i in range(0, 2 * n_xi + 1):
        xi_sig_hat[0:n_x, i] = rk_four(
            vehicle.f, xi_sig[0:n_x, i], xi_sig[n_x:n_xi, i], T
        )

    # Compute the mean and convariance from the transformed sigma points
    w_xi = 0.5 / (n_xi + kappa) * np.ones(2 * n_xi + 1)
    w_xi[0] = 2 * kappa * w_xi[0]
    xi = np.average(xi_sig_hat, axis=1, weights=w_xi)
    P_xi = np.cov(xi_sig_hat, ddof=0, aweights=w_xi)

    # Help to keep the covariance matrix symmetrical
    P_xi = (P_xi + np.transpose(P_xi)) / 2

    # Set the vehicle state estimates
    x_hat = xi[0:n_x]
    P_hat = P_xi[0:n_x, 0:n_x]

    # Find the number of observed features
    m_k = np.shape(a)[0]

    # Compute the a posteriori estimate if there are visible features
    if m_k > 0:

        # Compute a new set of sigma points using the latest x_hat and P_hat
        x_sig = np.zeros((n_x, 2 * n_x + 1))
        P_sig = np.linalg.cholesky((n_x + kappa) * P_hat)
        for i in range(0, n_x):
            x_sig[:, i + 1] = x_hat + P_sig[:, i]
            x_sig[:, n_x + i + 1] = x_hat - P_sig[:, i]

        # Find the expected measurement corresponding to each sigma point
        y_hat_sig = np.zeros((2 * m_k, 2 * n_x + 1))
        for j in range(0, 2 * n_x + 1):
            # Compute the expected measurements
            for i in range(0, m_k):
                y_hat_sig[2 * i, j] = np.sqrt(
                    (f_map[0, int(a[i])] - x_sig[0, j]) ** 2
                    + (f_map[1, int(a[i])] - x_sig[1, j]) ** 2
                )
                y_hat_sig[2 * i + 1, j] = (
                    np.unwrap(
                        [
                            np.arctan2(
                                f_map[1, int(a[i])] - x_sig[1, j],
                                f_map[0, int(a[i])] - x_sig[0, j],
                            )
                            - x_sig[2, j]
                        ]
                    )
                    - np.pi
                )

        # Recombine the sigma points
        w_x = 0.5 / (n_x + kappa) * np.ones(2 * n_x + 1)
        w_x[0] = 2 * kappa * w_x[0]
        y_hat = np.average(y_hat_sig, axis=1, weights=w_x)

        P_y = np.zeros((2 * m_k, 2 * m_k))
        P_xy = np.zeros((n_x, 2 * m_k))
        for i in range(0, 2 * n_x + 1):
            y_diff = y_hat_sig[:, i] - y_hat
            x_diff = x_sig[:, i] - x_hat
            P_y = P_y + w_x[i] * (y_diff.reshape((2 * m_k, 1))) @ np.transpose(
                y_diff.reshape((2 * m_k, 1))
            )
            P_xy = P_xy + w_x[i] * (x_diff.reshape((n_x, 1))) @ np.transpose(
                y_diff.reshape((2 * m_k, 1))
            )
        P_y = P_y + np.kron(np.identity(m_k), R)

        # Help to keep the covariance matrix symmetrical
        P_y = (P_y + np.transpose(P_y)) / 2

        # Update the estimate
        K = P_xy @ np.linalg.inv(P_y)
        x_hat = x_hat + K @ (y_m - y_hat)
        P_hat = P_hat - K @ P_y @ np.transpose(K)

        # Help keep the covariance matrix symmetric
        P_hat = (P_hat + np.transpose(P_hat)) / 2

    return x_hat, P_hat


# %%
# SIMULATE THE SYSTEM

# Set the covariance matrices
Q = np.diag([SIGMA_SPEED ** 2, SIGMA_SPEED ** 2])
R = np.diag([SIGMA_RANGE ** 2, SIGMA_BEARING ** 2])

# Initialize state, intput, and estimator variables
x = np.zeros((3, N))
v_m = np.zeros((2, N))
x_hat_UKF = np.zeros((3, N))
P_hat_UKF = np.zeros((3, 3, N))

# Initialize the state
x_init = np.zeros(3)

# Set the initial guess of the estimator
x_guess = x_init + np.array([5.0, -5.0, 0.1])
P_guess = np.diag(np.square([5.0, -5.0, 0.1]))

# Set the initial conditions
x[:, 0] = x_init
v_m[:, 0] = np.zeros(2)
x_hat_UKF[:, 0] = x_guess
P_hat_UKF[:, :, 0] = P_guess

KAPPA = 3 - np.shape(x)[0]

for i in range(1, N):

    # Compute some inputs (i.e., drive around)
    v = np.array([2.05, 1.95])

    # Run the vehicle motion model
    x[:, i] = rk_four(vehicle.f, x[:, i - 1], v, T)

    # Model the rate sensors
    v_m[0, i] = v[0] + np.sqrt(Q[0, 0]) * np.random.randn(1)
    v_m[1, i] = v[1] + np.sqrt(Q[1, 1]) * np.random.randn(1)

    # Run the measurement model
    y_m, a = RandB_sensor(x[:, i], f_map, R)

    # Run the UKF estimator
    x_hat_UKF[:, i], P_hat_UKF[:, :, i] = UKF(
        x_hat_UKF[:, i - 1],
        P_hat_UKF[:, :, i - 1],
        v_m[:, i - 1],
        y_m,
        a,
        f_map,
        Q,
        R,
        KAPPA,
    )

# %%
# PLOT THE SIMULATION OUTPUTS

# Find the scaling factors for covariance bounds
alpha = 0.01
s1 = chi2.isf(alpha, 1)
s2 = chi2.isf(alpha, 2)

# Set some plot limits for better viewing
x_range = 0.5
y_range = 0.5
theta_range = 0.1
phi_range = 0.1

# Plot the errors with covariance bounds
sigma = np.zeros((3, N))
fig1 = plt.figure(1)
ax1 = plt.subplot(311)
sigma[0, :] = np.sqrt(s1 * P_hat_UKF[0, 0, :])
plt.fill_between(t, -sigma[0, :], sigma[0, :], color="C0", alpha=0.2)
plt.plot(t, x[0, :] - x_hat_UKF[0, :], "C0")
plt.ylabel(r"$e_1$ [m]")
plt.setp(ax1, xticklabels=[])
ax1.set_ylim([-x_range, x_range])
ax2 = plt.subplot(312)
sigma[1, :] = np.sqrt(s1 * P_hat_UKF[1, 1, :])
plt.fill_between(t, -sigma[1, :], sigma[1, :], color="C0", alpha=0.2)
plt.plot(t, x[1, :] - x_hat_UKF[1, :], "C0")
plt.ylabel(r"$e_2$ [m]")
plt.setp(ax2, xticklabels=[])
ax2.set_ylim([-y_range, y_range])
ax3 = plt.subplot(313)
sigma[2, :] = np.sqrt(s1 * P_hat_UKF[2, 2, :])
plt.fill_between(t, -sigma[2, :], sigma[2, :], color="C0", alpha=0.2)
plt.plot(t, x[2, :] - x_hat_UKF[2, :], "C0")
plt.ylabel(r"$e_3$ [rad]")
plt.setp(ax3, xticklabels=[])
ax3.set_ylim([-theta_range, theta_range])
plt.xlabel(r"$t$ [s]")

# Plot the actual versus estimated positions on the map
fig2, ax = plt.subplots()
circle = Circle(x[0:2, 0], radius=R_MAX, alpha=0.2, color="C0", label="Sensing range")
ax.add_artist(circle)
plt.plot(x[0, :], x[1, :], "C0", label="Actual")
plt.plot(x_hat_UKF[0, :], x_hat_UKF[1, :], "C1--", label="Estimated")
plt.plot(f_map[0, :], f_map[1, :], "C2*", label="Transmitter")
plt.axis("equal")
ax.set_xlim([np.min(x_hat_UKF[0, :]) - 10, np.max(x_hat_UKF[0, :]) + 10])
ax.set_ylim([np.min(x_hat_UKF[1, :]) - 10, np.max(x_hat_UKF[1, :]) + 10])
plt.xlabel(r"$x_1$ [m]")
plt.ylabel(r"$x_2$ [m]")
plt.legend()

# Show the plot to the screen
plt.show()

# %%
