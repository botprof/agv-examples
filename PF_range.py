"""
Example PF_range.py
Author: Joshua A. Marshall <joshua.marshall@queensu.ca>
GitHub: https://github.com/botprof/agv-examples
"""

# %%
# SIMULATION SETUP

import numpy as np
import matplotlib.pyplot as plt
from mobotpy.integration import rk_four
from mobotpy.models import DiffDrive
from scipy.stats import chi2

# Set the simulation time [s] and the sample period [s]
SIM_TIME = 30.0
T = 0.5

# Create an array of time values [s]
t = np.arange(0.0, SIM_TIME, T)
N = np.size(t)

# %%
# VEHICLE SETUP

# Set the track length of the vehicle [m]
ELL = 1.0

# Create a vehicle object of type DiffDrive
vehicle = DiffDrive(ELL)

# Uncertainty in wheel speed measurements [m/s]
SIGMA_SPEED = 0.01

# Set the range [m] sensor uncertainty
SIGMA_RANGE = 0.3

# %%
# DEAD RECKONING EXAMPLE

# Set the number of particles to use
M = 1000

# Create an array of particles for each time index
x_pf = np.zeros((3, M, N))

# Set the covariance matrices
Q = np.diag([SIGMA_SPEED**2, SIGMA_SPEED**2])

# Initialize the vehicle's true state
x = np.zeros((3, N))

# Initialize a estimated pose estimate
x_hat = np.zeros((3, N))

# Initialize a covariance matrix
P_hat = np.zeros((3, 3, N))

# Set the initial process covariance
P_hat[:, :, 0] = np.diag(np.square([0.1, 0.1, 0.01]))

# Initialize the first particles on the basis of the initial uncertainty
for i in range(1, M):
    x_pf[:, i, 0] = x_hat[:, 0] + \
        np.sqrt(P_hat[:, :, 0]) @ np.random.standard_normal(3)

# Initialize the first particles on a uniform distribution over the space
# for i in range(1, M):
#     x_pf[:, i, 0] = 100 * np.random.uniform(-1, 1, 3)

for i in range(1, N):

    # Compute some inputs (i.e., drive around)
    v = np.array([3.95, 4.05])

    # Run the vehicle motion model
    x[:, i] = rk_four(vehicle.f, x[:, i - 1], v, T)

    # Propagate each particle through the motion model
    for j in range(0, M):

        # Model the proprioceptive sensors (i.e., speed and turning rate)
        v_m = v + np.sqrt(Q) @ np.random.standard_normal(2)

        # Propagate each particle
        x_pf[:, j, i] = rk_four(vehicle.f, x_pf[:, j, i - 1], v_m, T)

# Plot the results of the dead reckoning example
plt.figure(1)
plt.plot(x_pf[0, :, 0], x_pf[1, :, 0], ".", label="Particles", alpha=0.2)
for k in range(1, N, 1):
    plt.plot(x_pf[0, :, k], x_pf[1, :, k], ".", alpha=0.2)
plt.plot(x[0, :], x[1, :], "C0", label="Actual path")
plt.axis("equal")
plt.xlabel("$x$ [m]")
plt.ylabel("$y$ [m]")
plt.legend()
plt.show()

# %%
# CREATE A MAP OF FEATURES

# Set the number of features in the map
m = 50

# Set the size [m] of a square map
D_MAP = 100

# Create a map of randomly placed feature locations
f_map = np.zeros((2, m))
for i in range(0, m):
    f_map[:, i] = D_MAP * (2.0 * np.random.rand(2) - 1.0)

plt.figure(2)
plt.plot(f_map[0, :], f_map[1, :], "C2*", label="Feature")
plt.axis("equal")
plt.xlabel("$x$ [m]")
plt.ylabel("$y$ [m]")
plt.legend()
plt.show()

# %%
# FUNCTION TO MODEL RANGE TO FEATURES


def range_sensor(x, f_map, R, R_MAX, R_MIN):
    """Function to model the range sensor."""

    # Find the indices of features that are within range (R_MIN, R_MAX)
    a = np.array([])
    for i in range(0, m):
        r = np.sqrt((f_map[0, i] - x[0]) ** 2 + (f_map[1, i] - x[1]) ** 2)
        if np.all([r < R_MAX, r > R_MIN]):
            a = np.append(a, i)

    # Simulate for each time
    if np.shape(a)[0] > 0:

        # Assign the size of the output
        k = np.shape(a)[0]
        y = np.zeros(k)

        # Compute the range and bearing to all features (plus sensor noise)
        for i in range(0, k):

            # Range measurement [m]
            y[i] = (
                np.sqrt(
                    (f_map[0, int(a[i])] - x[0]) ** 2
                    + (f_map[1, int(a[i])] - x[1]) ** 2
                )
                + np.sqrt(R[0, 0]) * np.random.standard_normal()
            )

    else:

        # No features were found within the sensing range
        y = np.array([])

    # Return the measurements and indices map
    return y, a


# %%
# PARTICLE FILTER FUNCTIONS


def pf_resample(x_pf, x_likelihood):
    """Function to resample particles."""

    # Initialize a set of output particles
    x_pf_resampled = np.zeros((3, M))

    # Do the resampling (one way)
    indices = np.searchsorted(np.cumsum(x_likelihood),
                              np.random.random_sample(M))
    for j in range(0, M):
        x_pf_resampled[:, j] = x_pf[:, indices[j]]

    # Return the resampled particles
    return x_pf_resampled


def diffdrive_pf(x_pf, v, y, a, f_map, Q, R, T):
    """Particle filter for differential drive vehicle function."""

    # Find the number of observed features
    m_k = a.shape[0]

    # Initialize the output
    x_pf_new = np.zeros((3, M))

    # Propagate the particles through the vehicle model (i.e., a priori step)
    for j in range(0, M):

        # Model the wheel speed measurements
        v_m = v + np.sqrt(Q) @ np.random.standard_normal(2)

        # Propagate each particle
        x_pf_new[:, j] = rk_four(vehicle.f, x_pf[:, j], v_m, T)

    # Set likelihoods all equal to start the a posteriori step
    x_likelihood = 1.0 / M * np.ones(M)

    # Compute the relative likelihood
    if m_k > 1:

        # Set up some arrays
        y_hat = np.zeros((m_k, M))
        y_dif = np.zeros(m_k)

        # Compute some needed matrices
        R_inv = np.linalg.inv(np.kron(np.identity(m_k), R))
        R_det = np.linalg.det(np.kron(np.identity(m_k), R))

        for j in range(0, M):

            # For each visible beacon find the expected measurement
            for i in range(0, m_k):
                y_hat[i, j] = np.sqrt(
                    (f_map[0, int(a[i])] - x_pf_new[0, j]) ** 2
                    + (f_map[1, int(a[i])] - x_pf_new[1, j]) ** 2
                )

            # Compute the relative likelihoods
            y_dif = y - y_hat[:, j]
            x_likelihood[j] = (
                1.0
                / ((2.0 * np.pi) ** (m_k / 2) * np.sqrt(R_det))
                * np.exp(-0.5 * y_dif.T @ R_inv @ y_dif)
            )

        # Normalize the likelihoods
        x_likelihood /= np.sum(x_likelihood)

        # Generate a set of a posteriori particles by re-sampling on the basis of the likelihoods
        x_pf_new = pf_resample(x_pf_new, x_likelihood)

    return x_pf_new


# %%
# RUN THE PARTICLE FILTER SIMULATION

# Initialize some arrays
x_pf = np.zeros((3, M, N))
x_hat = np.zeros((3, N))
P_hat = np.zeros((3, 3, N))

# Initialize the initial guess to a location different from the actual location
x_hat[:, 0] = x[:, 0] + np.array([0, 5.0, 0.1])

# Set some initial conditions
P_hat[:, :, 0] = np.diag(np.square([5.0, 5.0, 0.1]))

# Set the covariance matrices
Q = np.diag([SIGMA_SPEED**2, SIGMA_SPEED**2])

# Set sensor range
R_MAX = 25.0
R_MIN = 1.0

# Set the range and bearing covariance
R = np.diag([SIGMA_RANGE**2])

# Initialize the first particles on the basis of the initial uncertainty
for i in range(1, M):
    x_pf[:, i, 0] = x_hat[:, 0] + \
        np.sqrt(P_hat[:, :, 0]) @ np.random.standard_normal(3)

# Initialize the first particles on the basis of the initial uncertainty
# for i in range(1, M):
#     x_pf[:, i, 0] = 100 * np.random.uniform(-1, 1, 3)

# Simulate for each time
for i in range(1, N):

    # Compute some inputs (i.e., drive around)
    v = np.array([3.95, 4.05])

    # Run the vehicle motion model
    x[:, i] = rk_four(vehicle.f, x[:, i - 1], v, T)

    # Run the range and bearing sensor model
    y_m, a = range_sensor(x[:, i], f_map, R, R_MAX, R_MIN)

    # Run the particle filter
    x_pf[:, :, i] = diffdrive_pf(x_pf[:, :, i - 1], v, y_m, a, f_map, Q, R, T)

# %%
# PLOT THE SIMULATION OUTPUTS

# Change some plot settings (optional)
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{cmbright,amsmath,bm}")
plt.rc("savefig", format="pdf")
plt.rc("savefig", bbox="tight")

# Plot the results of the particle filter simulation
plt.figure(3)
plt.plot(x_pf[0, :, 0], x_pf[1, :, 0], ".", label="Particles", alpha=0.2)
for k in range(1, N, 1):
    plt.plot(x_pf[0, :, k], x_pf[1, :, k], ".", alpha=0.2)
plt.plot(x[0, :], x[1, :], "C0", label="Actual path")
plt.plot(f_map[0, :], f_map[1, :], "C2*", label="Feature")
plt.axis("equal")
plt.xlabel("$x$ [m]")
plt.ylabel("$y$ [m]")
plt.legend()
plt.show()

# Compute the mean errors and estimated covariance bounds
for i in range(0, N):
    x_hat[0, i] = np.mean(x_pf[0, :, i])
    x_hat[1, i] = np.mean(x_pf[1, :, i])
    x_hat[2, i] = np.mean(x_pf[2, :, i])

for i in range(0, N):
    P_hat[:, :, i] = np.cov(x_pf[:, :, i])

# Find the scaling factors for plotting covariance bounds
ALPHA = 0.01
s1 = chi2.isf(ALPHA, 1)
s2 = chi2.isf(ALPHA, 2)

fig5 = plt.figure(4)
sigma = np.zeros((3, N))
ax1 = plt.subplot(311)
sigma[0, :] = np.sqrt(s1 * P_hat[0, 0, :])
plt.fill_between(t, -sigma[0, :], sigma[0, :], color="C1", alpha=0.2)
plt.plot(t, x[0, :] - x_hat[0, :], "C0")
plt.ylabel(r"$e_1$ [m]")
plt.setp(ax1, xticklabels=[])
ax2 = plt.subplot(312)
sigma[1, :] = np.sqrt(s1 * P_hat[1, 1, :])
plt.fill_between(t, -sigma[1, :], sigma[1, :], color="C1", alpha=0.2)
plt.plot(t, x[1, :] - x_hat[1, :], "C0")
plt.ylabel(r"$e_2$ [m]")
plt.setp(ax2, xticklabels=[])
ax3 = plt.subplot(313)
sigma[2, :] = np.sqrt(s1 * P_hat[2, 2, :])
plt.fill_between(t, -sigma[2, :], sigma[2, :], color="C1", alpha=0.2)
plt.plot(t, x[2, :] - x_hat[2, :], "C0")
plt.ylabel(r"$e_3$ [rad]")
plt.xlabel(r"$t$ [s]")
plt.show()

# %%
