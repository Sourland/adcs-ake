import numpy as np
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from scipy.optimize import least_squares, Bounds, basinhopping
import astropy.units as u
from skyfield.api import EarthSatellite, load
from sgp4.model import Satrec, WGS84
from utils import update_tle, jd_fr_to_tle_epoch, calculate_keplerian_elements

mu = 398600.4418  # gravitational parameter


def sgp4_propagate(line1, line2, ephemeris_times):
    # Initialize arrays to hold positions and velocities
    ts = load.timescale()
    ephemerides = []

    # Initialize a new Satrec object
    satrec = Satrec.twoline2rv(line1, line2, WGS84)

    # Create a Skyfield satellite from the Satrec object
    satellite = EarthSatellite.from_satrec(satrec, ts)

    # Propagate and save the position and velocity for each time in ephemeris_times
    for time in ephemeris_times:
        t = ts.utc(time)
        position, velocity, message = satellite._position_and_velocity_TEME_km(t)
        ephemerides.append(calculate_keplerian_elements(position, velocity))


    # Convert lists to arrays for easier manipulation
    ephemerides = np.array(ephemerides)

    return ephemerides


def residuals(initial_guess, line1, line2, ephemerides, ephemeris_times, current_time):
    """
    Calculates the residuals between the ephemeris data and the SGP4 propagated TLE data.

    Returns: residuals
    """
    epoch = jd_fr_to_tle_epoch(current_time.jd1, current_time.jd2)

    inclination, raan, eccentricity, arg_perigee, mean_anomaly, mean_motion = initial_guess
    line1, line2 = update_tle(line1, line2, inclination, raan, eccentricity, arg_perigee,
                              mean_anomaly, mean_motion, epoch)

    sgp4_data = sgp4_propagate(line1, line2, ephemeris_times)
    return np.mean((ephemerides - sgp4_data) ** 2)  # Replace with the actual calculation


def fit_tle_to_ephemeris(line1, line2, initial_guess, ephemerides, future_times, current_time):
    """
    Fits a TLE to ephemeris data using the least squares method.

    Returns: the best-fit TLE parameters
    """
    bounds = ([0, 0, 0, 0, 0, 1], [180, 360, 1, 360, 360, 16])

    minimizer_args = {"method": 'Powell',
                      "args": (line1, line2, ephemerides, future_times, current_time)}

    result = basinhopping(residuals, initial_guess, minimizer_kwargs=minimizer_args, niter=100)

    return result.x  # The optimized TLE parameters


def calculate_E(true_anomaly, eccentricity):
    true_anomaly = true_anomaly % (2 * np.pi)
    E = np.arccos((eccentricity + np.cos(true_anomaly)) / (1 + eccentricity * np.cos(true_anomaly)))
    if np.sin(true_anomaly) < 0:
        E = 2 * np.pi - E
    return E


def calculate_mean_anomaly(E, eccentricity):
    mean_anomaly = E - eccentricity * np.sin(E)
    mean_anomaly = mean_anomaly % (2 * np.pi)
    return mean_anomaly
