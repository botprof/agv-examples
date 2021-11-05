"""
Example oneD_dynamic_observer.py
Author: Joshua A. Marshall <joshua.marshall@queensu.ca>
GitHub: https://github.com/botprof/agv-examples
"""

# %% SIMULATION SETUP

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

# Set the simulation time [s] and the sample period [s]
SIM_TIME = 5.0
T = 0.1

# Create an array of time values [s]
t = np.arange(0, SIM_TIME, T)
N = np.size(t)

# %% VEHICLE MODEL DEFINTION

# Set the mass of the vehicle [kg]
M = 10.0

# Function that models the vehicle and sensor(s) in discrete time
F = np.array([[1, T], [0, 1]])
G = np.array([[T**2/(2*M)], [T/M]])
H = np.array([[1, 0]])

def vehicle(x, u, F, G):
    """Discrete-time 1D dynamic vehicle model."""
    x_new = F @ x + G @ [u]
    return x_new

# %% OBSERVER DEFINTION

# Choose estimator gains for stability
lambda_z = np.array([0.5, 0.4])
LT = signal.place_poles(F.T, H.T, lambda_z)

def observer(x_hat, u, y, F, G, H, L):
    """Observer-based state estimator."""
    x_hat_new = F @ x_hat + G @ [u] + L @ ([y] - H @ x_hat)
    return x_hat_new

# %% RUN SIMULATION

# Initialize arrays that will be populated with our inputs and states
x = np.zeros((2, N))
u = np.zeros(N)
x_hat = np.zeros((2, N))

# Set the initial position [m], velocity [m/s], and force input [N]
x[0, 0] = 1.0
x[1, 0] = 0.0
u[0] = 0.0

# Set the initial estimated position (different from the actual position)
x_hat[0, 0] = 0.0
x_hat[0, 0] = 0.0

# Run the simulation for time step k
for k in range(1, N):
    y = x[0, k-1]
    x_hat[:, k] = observer(x_hat[:, k-1], u[k-1], y, F, G, H,
                           LT.gain_matrix.T)
    x[:, k] = vehicle(x[:, k-1], u[k-1], F, G)
    u[k] = 2.0*np.sin(k*T)

# %% MAKE A PLOT

# Change some plot settings (optional)
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{cmbright,amsmath}')
plt.rc('savefig', format='pdf')

# Plot the states (x) and input (u) vs time (t)
fig1 = plt.figure(1)
ax1a = plt.subplot(211)
plt.plot(t, x[0, :], 'C0')
plt.step(t, x_hat[0, :], 'C1--', where='post')
plt.grid(color='0.95')
plt.ylabel(r'$x_1$ [m]')
plt.setp(ax1a, xticklabels=[])
plt.legend(['Actual', 'Estimated'])
ax1b = plt.subplot(212)
plt.plot(t, x[1, :], 'C0')
plt.step(t, x_hat[1, :], 'C1--', where='post')
plt.grid(color='0.95')
plt.ylabel(r'$x_2$ [m/s]')
plt.xlabel(r'$t$ [s]')

# Save the plot
plt.savefig('../agv-book/figs/ch2/oneD_dynamic_observer_fig1.pdf')

# %%

# Show all the plots to the screen
plt.show()
