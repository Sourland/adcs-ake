from collections import deque

from sgp4.model import Satrec, WGS84

from utils import *
from matplotlib import pyplot as plt
from skyfield.api import Loader, EarthSatellite
import numpy as np
from datetime import timedelta, datetime
from astropy.time import Time
from TLE import *
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
import astropy.units as u

# Load the TLE data
load = Loader('~/Documents/PeakSAT/aocs-sim')
ts = load.timescale()
mu = 398600.4418
# Example GPS groundtruth data (replace with your actual GPS data)
gps_positions = sim_cords  # List of GPS positions for each timestep
gps_velocities = sim_velocities  # List of GPS velocities for each timestep
# Initial TLE lines
line1 = "1 99999U 00000    24183.37500000  .00000000  00000-0  00000-0 0 00001"
line2 = "2 99999 097.4805 265.0320 0011464 250.4455 109.5401 15.26449498000012"

year, month, day, hour, minute, second = 2024, 7, 1, 9, 0, 0


class OrbitChecker:
    def __init__(self, mean_motion):
        # Convert mean motion from revs/day to revs/sec
        self.mean_motion_sec = mean_motion / (24 * 60 * 60)

        # Calculate the period of the orbit (in seconds)
        self.T = 1 / self.mean_motion_sec

        # Initialize the time of the last completed orbit to None
        self.last_completed = None

    def check_orbit(self, t):
        # Calculate the number of orbits completed
        completed_orbits = int(t / self.T)

        if self.last_completed is None:
            # If this is the first check, record the number of completed orbits
            self.last_completed = completed_orbits
            return completed_orbits > 0
        else:
            # If this is not the first check, see if any new orbits have been completed
            new_orbits = completed_orbits - self.last_completed
            self.last_completed = completed_orbits
            return new_orbits > 0


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

satrec = Satrec.twoline2rv(line1, line2, WGS84)
satellite = EarthSatellite.from_satrec(satrec, ts)
observations = deque(maxlen=10)
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
        epoch = jd_fr_to_tle_epoch(start_date_t.jd1, start_date_t.jd2)
        # estimated_pos, estimates_vel = differential_correction(np.hstack((position, velocity)), np.array(observations),
        #                                                        np.arange(10), mu=398600.4418)
        GPS_POS = gps_position * u.km
        GPS_VEL = gps_velocity * u.km / u.s

        orbit = Orbit.from_vectors(Earth, GPS_POS, GPS_VEL, epoch=start_date_t)
        semi_major_axis = orbit.a.value
        mean_motion = np.sqrt(mu / semi_major_axis ** 3) * (24 * 60 * 60) / (2 * np.pi)
        true_anomaly = orbit.nu.value
        eccentricity = orbit.ecc.value

        true_anomaly = true_anomaly % (2 * np.pi)
        E = np.arccos((eccentricity + np.cos(true_anomaly)) / (1 + eccentricity * np.cos(true_anomaly)))
        if np.sin(true_anomaly) < 0:
            E = 2 * np.pi - E

        mean_anomaly = E - eccentricity * np.sin(E)
        mean_anomaly = mean_anomaly % (2 * np.pi)
        arg_perigee = orbit.argp.value
        raan = orbit.raan.value
        inclination = orbit.inc.value

        orbit_checker = OrbitChecker(mean_motion)
        has_completed_orbit = orbit_checker.check_orbit(step)

        line1, line2 = update_tle(line1, line2, inclination, raan, eccentricity, arg_perigee,
                                  mean_anomaly, mean_motion, epoch, has_completed_orbit)
        satrec = Satrec.twoline2rv(line1, line2, WGS84)
        satellite = EarthSatellite.from_satrec(satrec, ts)
        # observations = deque(maxlen=120)
        # print("TLE UPDATED")
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
