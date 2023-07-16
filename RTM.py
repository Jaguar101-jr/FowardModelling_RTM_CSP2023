
## Forward modeling for seismic wave propagation using the acoustic wave equation
## for acoustic RTM
## Centre for Seismological Phenomena - Sudan 2023.

#In terms of boundary conditions, the code assumes the following:

#Spatial Boundary Conditions: The code uses a simple "free-surface" boundary condition,
##assuming that there are no reflective boundaries at the edges of the model. This implies
##that waves can propagate freely through the edges of the model without being reflected
##or affected by boundary conditions. It assumes an infinite extent in the horizontal
##dimensions, allowing the wave to propagate outside the defined model boundaries.

#Source Boundary Conditions: The code assumes a point source at a specific location
##within the model grid. At each time step, the source waveform is applied at this source
##location, representing the initiation of the seismic wave. The code does not consider
##complex source conditions or distributed sources.


import numpy as np
import matplotlib.pyplot as plt


def forward_modeling(velocity_model, time_steps, dx, dz, dt):
    # Number of grid points in x and z directions
    nz, nx = velocity_model.shape

    # Initialize the wavefield at t = 0 and t = -1
    current_wavefield = np.zeros((nz, nx))
    previous_wavefield = np.zeros((nz, nx))

    # Calculate the square of the velocity model
    velocity_sq = velocity_model ** 2

    # Initialize the seismic data array
    seismic_data = np.zeros((time_steps, nz, nx))

    # Forward modeling loop
    for t in range(time_steps):
        # Apply the acoustic wave equation
        next_wavefield = 2 * current_wavefield - previous_wavefield \
                         + velocity_sq * (dt ** 2) * (
                                 np.gradient(np.gradient(current_wavefield, dx, axis=1), dx, axis=1)
                                 + np.gradient(np.gradient(current_wavefield, dz, axis=0), dz, axis=0)
                         )

        # Update the previous and current wavefields
        previous_wavefield = current_wavefield.copy()
        current_wavefield = next_wavefield.copy()

        # Store the current wavefield as seismic data
        seismic_data[t] = current_wavefield

    return seismic_data


# Example usage
# Define the velocity model for two layers
nz, nx = 100, 100
velocity_model1 = np.ones((nz, nx)) * 2000  # Velocity for layer 1
velocity_model2 = np.ones((nz, nx)) * 3000  # Velocity for layer 2

# Parameters for grid spacing and time step
dx = 1.0  # Grid spacing in x direction
dz = 1.0  # Grid spacing in z direction
dt = 0.001  # Time step size
time_steps = 100

# Perform forward modeling
seismic_data = forward_modeling(velocity_model1, time_steps, dx, dz, dt)

# Display the seismic data at a specific time step
timestep = 50  # Choose a time step to visualize
plt.imshow(seismic_data[timestep], cmap='gray')
plt.colorbar()
plt.title(f"Seismic Data at Time Step {timestep}")
plt.show()
