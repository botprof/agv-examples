"""
Example usv_demo_3dof.py
Original author: Joshua A. Marshall <joshua.marshall@queensu.ca>
Modified by: Thomas M. C. Sears <thomas.sears@queensu.ca>
GitHub: https://github.com/botprof/agv-examples

This code demonstrates basic use of the 'models.LongitudinalUSV' class
to simulate a 3DOF (surge [x], heave [z], pitch).

Run me in interactive Python for an animation!
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
# Create a space-time wave model (simple wave)
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
# BOAT STATE (SURGE, HEAVE, PITCH)


# Start with all zeros
x = np.zeros((3, N))

# Keep surge (x) at zero. Set heave and pitch to wave
# height and slope respectively
x[1, :], x[2, :] = wave(x[0, :], t)

# %%
# WAVE SIMULATION

# Define spatial range to plot wave
x_range = np.arange(-20, 20, 0.1)
N_x = np.size(x_range)

# Initialize array to store wave heights for all N time steps
wave_heights = np.zeros((N_x, N))

for i in range(N):
    wave_heights[:, i], _ = wave(x_range, t[i])


# %%
# MAKE AN ANIMATION

# Set the side length of the vehicle [m]
LENGTH = 2.0

# Use the LongitudinalUSV class to create an animation
vehicle = LongitudinalUSV(LENGTH)

# Create and save the animation
ani = vehicle.animate(x, T,
                      wave_positions=x_range, wave_data=wave_heights,
                      relative=False)

# %%

# Show all the plots to the screen
plt.show()

# Show animation in HTML output if you are using IPython or Jupyter notebooks
plt.rc('animation', html='jshtml')
display(ani)
plt.close()
