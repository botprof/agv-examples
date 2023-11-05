"""
Example UT_example.py
Author: Joshua A. Marshall <joshua.marshall@queensu.ca>
GitHub: https://github.com/botprof/agv-examples
"""

# %%
# SIMULATION SETUP

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from matplotlib import patches

# Set the number of points to simulate for ground truth
N = 100000

# Initialized input and output arrays
x = np.zeros((2, N))
y = np.zeros((2, N))

# Set the mean and covariance values for the inputs
R_BAR = 1.0
THETA_BAR = np.pi / 2.0
SIGMA_R = 0.02
SIGMA_THETA = 15.0

# Generate random inputs for ground truth
x[0, :] = R_BAR + SIGMA_R * np.random.randn(1, N)
x[1, :] = THETA_BAR + SIGMA_THETA * np.pi / 180.0 * np.random.randn(1, N)

# %%
# FUNCTION DEFINITIONS

# The nonlinear polar to rectilinear transformation
def h(x):
    y = np.zeros(2)
    y[0] = x[0] * np.cos(x[1])
    y[1] = x[0] * np.sin(x[1])
    return y


# Function that implements the unscented transformation
def unscented_transform(f, x, P_x, kappa):
    """Unscented transform of statistics."""

    # Get the dimension of the random variable
    n = np.shape(x)[0]

    # Create array for sigma points
    x_sig = np.zeros((n, 2 * n + 1))

    # Find matrix square root
    nP_sig = np.linalg.cholesky((n + kappa) * P_x)

    # Generate the sigma points
    x_sig[:, 0] = x
    for i in range(0, n):
        x_sig[:, i + 1] = x + nP_sig[:, i]
        x_sig[:, n + i + 1] = x - nP_sig[:, i]

    # Pass sigma points through the system model
    y_sig = np.zeros((n, 2 * n + 1))
    for i in range(0, 2 * n + 1):
        y_sig[:, i] = f(x_sig[:, i])

    # Compute weighted mean and covariance from the transformed sigma points
    w = 0.5 / (n + kappa) * np.ones(2 * n + 1)
    w[0] = 2 * kappa * w[0]
    y = np.average(y_sig, axis=1, weights=w)
    P_y = np.cov(y_sig, ddof=0, aweights=w)

    # Help to keep the covariance matrix symmetrical
    P_y = (P_y + np.transpose(P_y)) / 2

    # Return the output mean an covariance
    return y, P_y


# %%
# COMPUTE AND PLOT GROUND TRUTH

# Compute outputs by passing each point through the nonlinear transformation
for i in range(0, N):
    y[:, i] = h(x[:, i])

# Compute the output statistics by brute force (using NumPy functions)
y_bar = np.mean(y, axis=1)
P_y = np.cov(y, ddof=0)

# Change some plot settings (optional)
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{cmbright,amsmath,bm}")
plt.rc("savefig", format="pdf")
plt.rc("savefig", bbox="tight")

# Plot the output
fig1, ax1 = plt.subplots()
plt.plot(y[0, :], y[1, :], "C0+", alpha=0.2)
plt.xlabel(r"$y_1$")
plt.ylabel(r"$y_2$")
plt.grid(color="0.95")
OUTPUT_STATS = (
    r"\begin{align*}"
    + r"N &="
    + str(N)
    + r"\\"
    + r"\bar{y}_1 &="
    + str(np.around(y_bar[0], decimals=3))
    + r"\\"
    + r"\bar{y}_2 &="
    + str(np.around(y_bar[1], decimals=3))
    + r"\\"
    + r"\end{align*}"
)
plt.text(0.05, 0.9, OUTPUT_STATS, transform=ax1.transAxes)

# Save the plot
plt.savefig("../agv-book/figs/ch5/UT_example_fig1.pdf")

# Show the plot to the screen
plt.show()

# %%
# COMPUTE THE OUTPUT STATISTICS BY LINEAR APPROXIMATION

# Find the scaling factor for plotting covariance bounds
ALPHA = 0.05
s2 = chi2.isf(ALPHA, 2)

# Create the plot and axes
fig2, ax2 = plt.subplots()
plt.xlabel(r"$y_1$")
plt.ylabel(r"$y_2$")
plt.grid(color="0.95")

# Create the 95 % confidence ellipse for the actual statistics
ell_actual = patches.Ellipse(
    (y_bar[0], y_bar[1]),
    2 * np.sqrt(s2 * P_y[0, 0]),
    2 * np.sqrt(s2 * P_y[1, 1]),
    angle=0,
    alpha=0.2,
    color="C0",
)
ax2.add_artist(ell_actual)

# Create the 95 % confidence ellipse for the linearized estimate
ell_linear = patches.Ellipse(
    (0, 1),
    2 * np.sqrt(s2) * 15.0 * np.pi / 180.0,
    2 * np.sqrt(s2) * 0.02,
    angle=0,
    alpha=0.2,
    color="C1",
)
ax2.add_artist(ell_linear)

# Plot location of means
plt.plot(y_bar[0], y_bar[1], "C0+")
plt.plot(0, 1, "C1+")

# Set the axis limits based on the actual covariance
ax2.set_xlim(y_bar[0] - np.sqrt(s2 * P_y[0, 0]), y_bar[0] + np.sqrt(s2 * P_y[0, 0]))
ax2.set_ylim(y_bar[1] - np.sqrt(s2 * P_y[1, 1]), y_bar[1] + np.sqrt(s2 * P_y[1, 1]))

# Add a legend and show the plot
plt.legend(["Actual", "Linearized"])

# Save the plot
plt.savefig("../agv-book/figs/ch5/UT_example_fig2.pdf")

# Show the plot to the screen
plt.show()

# %%
# COMPUTE THE OUTPUT STATISTICS BY THE UT

KAPPA = 3 - np.shape(x)[0]

y_u, P_u = unscented_transform(
    h,
    np.array([R_BAR, THETA_BAR]),
    np.diag([SIGMA_R ** 2, (SIGMA_THETA * np.pi / 180.0) ** 2]),
    KAPPA,
)

# Create the plot and axes
fig3, ax3 = plt.subplots()
plt.xlabel(r"$y_1$")
plt.ylabel(r"$y_2$")
plt.grid(color="0.95")

# Create a covariance ellipse for the actual statistics
ell_actual = patches.Ellipse(
    (y_bar[0], y_bar[1]),
    2 * np.sqrt(s2 * P_y[0, 0]),
    2 * np.sqrt(s2 * P_y[1, 1]),
    angle=0,
    alpha=0.2,
    color="C0",
)
ax3.add_artist(ell_actual)

# Create the 95 % confidence ellipse for the linearized estimate
ell_linear = patches.Ellipse(
    (0, 1),
    2 * np.sqrt(s2) * 15.0 * np.pi / 180.0,
    2 * np.sqrt(s2) * 0.02,
    angle=0,
    alpha=0.2,
    color="C1",
)
ax3.add_artist(ell_linear)

# Create a covariance ellipse for the unscented transform of statistics
ell_UT = patches.Ellipse(
    (y_u[0], y_u[1]),
    2 * np.sqrt(s2 * P_u[0, 0]),
    2 * np.sqrt(s2 * P_u[1, 1]),
    angle=0,
    alpha=0.2,
    color="C2",
)
ax3.add_artist(ell_UT)

# Plot location of means
plt.plot(y_bar[0], y_bar[1], "C0+")
plt.plot(0, 1, "C1+")
plt.plot(y_u[0], y_u[1], "C2+")

# Set the axis limits based on the actual covariance
ax3.set_xlim(y_bar[0] - np.sqrt(s2 * P_y[0, 0]), y_bar[0] + np.sqrt(s2 * P_y[0, 0]))
ax3.set_ylim(y_bar[1] - np.sqrt(s2 * P_y[1, 1]), y_bar[1] + np.sqrt(s2 * P_y[1, 1]))

# Add a legend and show the figure
plt.legend(["Actual", "Linearized", "Unscented"])

# Save the plot
plt.savefig("../agv-book/figs/ch5/UT_example_fig3.pdf")

# Show the plot to the screen
plt.show()
