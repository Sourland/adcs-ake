from sgp4.api import Satrec
import numpy as np
from astropy.time import Time
from utils import *
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
from collections import deque
from filterpy.kalman import ExtendedKalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise

line1 = "1 99999U 00000    24183.37500000  .00000000  00000-0  00000-0 0 00001"
line2 = "2 99999 097.4805 265.0320 0011464 250.4455 109.5401 15.26449498000012"

# Create a Satrec object from the initial TLE


year, month, day, hour, minute, second = 2024, 7, 1, 9, 0, 0
# Define the start date and time for the simulation
start_date = datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second, tzinfo=pytz.UTC)
start_date_t = Time(start_date, format='datetime', scale='utc')
# Define the time step for the simulation (in seconds)
time_step = 1
# N = sim_cords.shape[0]
N = 100000
# Define the number of steps for the simulation
gps_positions = sim_cords  # List of GPS positions for each timestep
gps_velocities = sim_velocities  # List of GPS velocities for each timestep
satellite = Satrec.twoline2rv(line1, line2)
positions = []
errors = []
# Simulate the orbit
observations = deque(maxlen=1000)
position_covariance = np.diag([0.447 * 1e-3, 0.447 * 1e-3, 0.447 * 1e-3])
velocity_covariance = np.diag([0.063 * 1e-3, 0.063 * 1e-3, 0.063 * 1e-3])

initial_covariance = np.block([[position_covariance, np.zeros((3, 3))],
                               [np.zeros((3, 3)), velocity_covariance]])

TLE_updated = False
for step in range(1, N):
    if step % 20000 == 0:
        print(f"STEP {step}")
    # Calculate the current time
    # Get the position and velocity from GPS in ECEF frame
    gps_position = gps_positions[step, :]
    gps_velocity = gps_velocities[step, :]
    observations.append(np.hstack((gps_position, gps_velocity)))
    # Generate a TLE from GPS position and velocity in ECEF frame
    # generated_tle_line1, generated_tle_line2 = generate_tle_from_gps(gps_position_ecef, gps_velocity_ecef)

    # Update the Satrec object with the generated TLE

    # Get the position and velocity of the satellite at the current time
    _, position, velocity = satellite.sgp4(start_date_t.jd1, start_date_t.jd2)
    positions.append(position)
    error = calculate_error(position, gps_position)

    if error > 10:
        points = MerweScaledSigmaPoints(6, alpha=0.1, beta=2., kappa=3)
        kf = UnscentedKalmanFilter(dim_x=6, dim_z=6, dt=time_step, fx=motion_model, hx=measurement_model, points=points)
        kf.x = np.hstack((gps_position, gps_velocity))
        # initialize the state covariance
        kf.P = np.diag(np.hstack((gps_position, gps_velocity)))
        # create the process noise matrix
        kf.Q = np.eye(6) * 0.01

        # create the measurement noise matrix
        kf.R = np.eye(3) * 0.1
        for observation in observations:
            kf.predict()
            kf.update(observation, measurement_jacobian, measurement_model)
        kf.predict()
        estimated_state = kf.x
        epoch = jd_fr_to_tle_epoch(start_date_t.jd1, start_date_t.jd2)
        inclination, raan, eccentricity, arg_perigee, mean_anomaly, mean_motion = calculate_keplerian_elements(
            gps_position, gps_velocity)
        line1, line2 = update_tle(line1, line2, inclination, raan, eccentricity, arg_perigee, mean_anomaly, mean_motion,
                                  epoch)
        satellite = Satrec.twoline2rv(line1, line2)
        TLE_updated = True
        # print(error)
    # if TLE_updated:
    #     _, position, velocity = satellite.sgp4(start_date_t.jd1, start_date_t.jd2)
    #     error = calculate_error(position, gps_position)
    #     TLE_updated = False
    #     errors.append(error)
    # else:
        errors.append(error)
    start_date = start_date + timedelta(seconds=time_step)
    start_date_t = Time(start_date, format='datetime', scale='utc')

positions = np.array(positions)
fig, axis = plt.subplots(3)
axis[0].plot(positions[:, 0])
axis[0].grid()
axis[0].set_xlabel("Time(seconds)")
axis[0].set_ylabel("Position x (km)")

axis[1].plot(positions[:, 1])
axis[1].grid()
axis[1].set_xlabel("Time(seconds)")
axis[1].set_ylabel("Position y (km)")

axis[2].plot(positions[:, 2])
axis[2].grid()
axis[2].set_xlabel("Time(seconds)")
axis[2].set_ylabel("Position z (km)")
plt.show()

velocities = np.array(gps_positions)
fig, axis = plt.subplots(3)
axis[0].plot(velocities[:, 0])
axis[1].plot(velocities[:, 1])
axis[2].plot(velocities[:, 2])
plt.show()

all_errors = np.array(errors)
plt.plot(all_errors)
plt.title("Orbital Knowledge Error")
plt.grid()
plt.xlabel("Time(seconds)")
plt.ylabel("Error (km)")
plt.show()

plt.hist(all_errors, bins=7)
plt.show()
