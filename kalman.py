from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise
import numpy as np
from main import hpop

# Define past position and velocity observations until timestep n-1
past_observations = np.array([hpop[0, :], hpop[1, :], hpop[2, :], hpop[3, :], hpop[4, :], hpop[5, :]])

dt = 1.0  # Time step

# Create the EKF instance
ekf = ExtendedKalmanFilter(dim_x=6, dim_z=6)

# Set the motion model, measurement model, and measurement Jacobian functions
ekf.f = motion_model
ekf.h = measurement_model
ekf.HJacobian = measurement_jacobian

# Define the initial state and covariance matrix
# Define the separate covariance matrices for position and velocity
position_covariance = np.diag([0.447 * 1e-3, 0.447 * 1e-3, 0.447 * 1e-3])
velocity_covariance = np.diag([0.063 * 1e-3, 0.063 * 1e-3, 0.063 * 1e-3])
initial_state = np.array(hpop[6, :])
initial_covariance = np.block([[position_covariance, np.zeros((3, 3))],
                               [np.zeros((3, 3)), velocity_covariance]])
ekf.x = initial_state
ekf.P = initial_covariance

# Define process noise covariance matrix Q
process_noise_cov = Q_discrete_white_noise(dim=4, dt=dt, var=0.01)

# Define measurement noise covariance matrix R
measurement_noise_cov = np.eye(6) * 0.1



# Predict the state at timestep n
ekf.predict()
# Access the estimated state at timestep n
estimated_state = ekf.x

print("Estimated state at timestep n:", estimated_state)
