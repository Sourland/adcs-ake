from datetime import datetime, timedelta
from collections import deque

from matplotlib import pyplot as plt

from sgp4.model import Satrec, WGS84
from skyfield.api import Loader

import astropy.units as u
from poliastro.bodies import Earth
from poliastro.twobody import Orbit

from utils import *
from TLE import fit_tle_to_ephemeris

# Constants and initial data
load = Loader('~/Documents/PeakSAT/aocs-sim')
ts = load.timescale()
mu = 398600.4418
gps_positions = sim_cords
gps_velocities = sim_velocities
line1 = "1 99999U 00000    24183.37500000  .00000000  00000-0  00000-0 0 00001"
line2 = "2 99999 097.4805 265.0320 0011464 250.4455 109.5401 15.26449498000012"
start_date = datetime.datetime(year=2024, month=7, day=1, hour=9, minute=0, second=0, tzinfo=pytz.UTC)
future_ephemerides_count = 1000


# Class definition
class OrbitChecker:
    def __init__(self, mean_motion):
        self.mean_motion_sec = mean_motion / (24 * 60 * 60)
        self.T = 1 / self.mean_motion_sec
        self.last_completed = None

    def check_orbit(self, t):
        completed_orbits = int(t / self.T)
        if self.last_completed is None:
            self.last_completed = completed_orbits
            return completed_orbits > 0
        else:
            new_orbits = completed_orbits - self.last_completed
            self.last_completed = completed_orbits
            return new_orbits > 0


# Prepare for orbit propagation
N = 40000
positions, velocities, all_errors = [], [], []
start_date_t = Time(start_date, format='datetime', scale='utc')
t = ts.utc(start_date)
satrec = Satrec.twoline2rv(line1, line2, WGS84)
satellite = EarthSatellite.from_satrec(satrec, ts)
observations = deque(maxlen=10)

# Orbit propagation
for step in range(N):
    gps_position = gps_positions[step, :]
    gps_velocity = gps_velocities[step, :]
    observations.append(np.hstack((gps_position, gps_velocity)))

    position, velocity, message = satellite._position_and_velocity_TEME_km(t)
    positions.append(position)
    velocities.append(velocity)

    error = calculate_error(position, gps_position)
    all_errors.append(error)

    if error > 1:
        future_times = [start_date + timedelta(seconds=i) for i in range(0, future_ephemerides_count)]
        if step + future_ephemerides_count > gps_positions.shape[0]:
            ephemeris_pos = gps_positions[step:, :]
            ephemeris_vel = gps_velocities[step:, :]
        else:
            ephemeris_pos = gps_positions[step:step + future_ephemerides_count, :]
            ephemeris_vel = gps_velocities[step:step + future_ephemerides_count, :]

        keplerian_ephemerides = []

        for i in range(len(future_times)):
            keplerian_elements = calculate_keplerian_elements(ephemeris_pos[i, :], ephemeris_vel[i, :])
            keplerian_ephemerides.append(keplerian_elements)

        keplerian_ephemerides = np.array(keplerian_ephemerides)
        initial_guess = keplerian_ephemerides[0, :]

        result = fit_tle_to_ephemeris(line1, line2, initial_guess, keplerian_ephemerides, future_times, start_date_t)
        inclination, raan, eccentricity, arg_perigee, mean_anomaly, mean_motion = result[0], result[1], result[2], \
                                                                                result[3], result[4], result[5]

        epoch = jd_fr_to_tle_epoch(start_date_t.jd1, start_date_t.jd2)
        line1, line2 = update_tle(line1, line2, inclination, raan, eccentricity, arg_perigee, mean_anomaly, mean_motion, epoch)
        satrec = Satrec.twoline2rv(line1, line2, WGS84)
        satellite = EarthSatellite.from_satrec(satrec, ts)

    start_date += timedelta(seconds=1)
    start_date_t = Time(start_date, format='datetime', scale='utc')
    t = ts.utc(start_date)

    # Print the current step and position
    if (step + 1) % 10000 == 0:
        print(f"Step {step + 1}: Position: {position}, Error: {error}")

# Plot error
plt.plot(all_errors)
plt.show()
