from collections import deque
from utils import *
from matplotlib import pyplot as plt
from skyfield.api import Loader, EarthSatellite
import numpy as np
from datetime import timedelta, datetime
from astropy.time import Time

# Load the TLE data
load = Loader('~/Documents/PeakSAT/aocs-sim')
ts = load.timescale()

# Example GPS groundtruth data (replace with your actual GPS data)
gps_positions = sim_cords  # List of GPS positions for each timestep
gps_velocities = sim_velocities  # List of GPS velocities for each timestep
# Initial TLE lines
line1 = "1 99999U 00000    24183.37500000  .00000000  00000-0  00000-0 0 00001"
line2 = "2 99999 097.4805 265.0320 0011464 250.4455 109.5401 15.26449498000012"

year, month, day, hour, minute, second = 2024, 7, 1, 9, 0, 0

# Number of steps
N = sim_cords.shape[0]  # Replace with the desired number of steps
N = 40000

positions = []
velocities = []
all_errors = []
# Propagate the orbit and update TLE if error exceeds threshold
start_date = datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second, tzinfo=pytz.UTC)
start_date_t = Time(
    start_date,
    format='datetime',
    scale='utc'
)
t = ts.utc(
    start_date
)

satellite = EarthSatellite(line1, line2, "Peaksat", ts)
observations = deque(maxlen=120)
for step in range(N):
    # Get the current GPS position and velocity
    gps_position = gps_positions[step, :]
    gps_velocity = gps_velocities[step, :]
    observations.append(np.hstack((gps_position, gps_velocity)))
    # Create a satellite object with the current TLE

    # Propagate the satellite orbit with a 10s timestep
    position, velocity, message = satellite._position_and_velocity_TEME_km(t)
    positions.append(position)
    velocities.append(velocity)
    # Calculate the knowledge error
    error = calculate_error(position, gps_position)
    all_errors.append(error)
    if error > 1:
        observations_mean = np.mean(observations, axis=0)
        epoch = jd_fr_to_tle_epoch(start_date_t.jd1, start_date_t.jd2)
        inclination, raan, eccentricity, arg_perigee, mean_anomaly, mean_motion = calculate_keplerian_elements(
            gps_position, gps_velocity)
        # mean_inclination, mean_raan, mean_eccentricity, mean_arg_perigee, mean_mean_anomaly, mean_mean_motion = \
        #     calculate_keplerian_elements(observations_mean[:3], observations_mean[3:])

        line1, line2 = update_tle(line1, line2, inclination, raan, eccentricity, arg_perigee,
                                  mean_anomaly, mean_motion, epoch)
        satellite = EarthSatellite(line1, line2, "Peaksat", ts)
        # observations = deque(maxlen=120)
        print("TLE UPDATED")
    # Re-propagate the satellite orbit with the updated TLE
    # position, velocity = satellite.at(t).position.km, satellite.at(t).velocity.km_per_s

    start_date += timedelta(seconds=1)
    start_date_t = Time(
        start_date,
        format='datetime',
        scale='utc'
    )
    t = ts.utc(
        start_date
    )
    # Print the current step and position
    if (step + 1) % 10000 == 0:
        print(f"Step {step + 1}: Position: {position}, Error: {error}")

    # Perform other calculations or operations as needed for each step

all_errors = np.array(all_errors)
# all_errors[all_errors > 100] = 1
plt.plot(all_errors)
plt.show()
