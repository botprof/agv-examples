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
N = 10000

# Initialized input and output arrays
x = np.zeros((2, N))
y = np.zeros((2, N))

# Set the mean and covariance values for the inputs
r_bar = 1.0
theta = np.pi / 2.0
sigma_r = 0.02
sigma_theta = 15.0

# Generate random inputs for ground truth
x[0, :] = r_bar + sigma_r * np.random.randn(1, N)
x[1, :] = theta + sigma_theta * np.pi / 180.0 * np.random.randn(1, N)

# %%
# FUNCTION DEFINITIONS

# The nonlinear polar to rectilinear transformation
def h(x):
    y = np.zeros(2)
    y[0] = x[0] * np.cos(x[1])
    y[1] = x[0] * np.sin(x[1])
    return y


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
output_stats = (
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
plt.text(0.05, 0.9, output_stats, transform=ax1.transAxes)

# Save the plot
plt.savefig("../agv-book/figs/ch5/UT_example_fig1.pdf")

# Show the plot to the screen
plt.show()

# %%
# COMPUTE THE OUTPUT STATISTICS BY LINEAR APPROXIMATION

# Find the scaling factor for plotting covariance bounds
alpha = 0.05
s2 = chi2.isf(alpha, 2)

# Create the plot and axes
fig2, ax2 = plt.subplots()
plt.xlabel(r"$y_1$")
plt.ylabel(r"$y_2$")

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

# Set up the input sigma points
x_sig = np.zeros((2, 4))
x_sig[:, 0] = np.array([1 + np.sqrt(2 * 0.02 ** 2), np.pi / 2])
x_sig[:, 1] = np.array([1, np.pi / 2 + np.sqrt(2 * (15 * np.pi / 180) ** 2)])
x_sig[:, 2] = np.array([1 - np.sqrt(2 * 0.02 ** 2), np.pi / 2])
x_sig[:, 3] = np.array([1, np.pi / 2 - np.sqrt(2 * (15 * np.pi / 180) ** 2)])

# Pass the sigma points through the nonlinearity
y_sig = np.zeros((2, 4))
for k in range(0, 4):
    y_sig[:, k] = h(x_sig[:, k])

# Approximate the mean and covariance of the output using the UT
y_UT_bar = np.mean(y_sig, axis=1)
P_UT = np.cov(y_sig, ddof=0)

# Create the plot and axes
fig3, ax3 = plt.subplots()
plt.xlabel(r"$y_1$")
plt.ylabel(r"$y_2$")

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
    (y_UT_bar[0], y_UT_bar[1]),
    2 * np.sqrt(s2 * P_UT[0, 0]),
    2 * np.sqrt(s2 * P_UT[1, 1]),
    angle=0,
    alpha=0.2,
    color="C2",
)
ax3.add_artist(ell_UT)

# Plot location of means
plt.plot(y_bar[0], y_bar[1], "C0+")
plt.plot(0, 1, "C1+")
plt.plot(y_UT_bar[0], y_UT_bar[1], "C2+")

# Set the axis limits based on the actual covariance
ax3.set_xlim(y_bar[0] - np.sqrt(s2 * P_y[0, 0]), y_bar[0] + np.sqrt(s2 * P_y[0, 0]))
ax3.set_ylim(y_bar[1] - np.sqrt(s2 * P_y[1, 1]), y_bar[1] + np.sqrt(s2 * P_y[1, 1]))

# Add a legend and show the figure
plt.legend(["Actual", "Linearized", "Unscented"])

# Save the plot
plt.savefig("../agv-book/figs/ch5/UT_example_fig3.pdf")

# Show the plot to the screen
plt.show()
