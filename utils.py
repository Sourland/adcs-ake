import math
from datetime import timedelta, datetime
import datetime
import numpy as np
import pandas as pd
import pytz
from astropy.time import Time

hpop = np.genfromtxt("hpopdate.txt", dtype=str,
                     encoding=None, delimiter=",").astype(float)
# hpop.columns = sgp4.columns

hpop_coordinates = hpop[:, 0:3]

hpop_velocities = hpop[:, 3:]

st_d_cords = 0.447
mu_cords = 1.002
st_d_velocity = 0.063
mu_velocity = 0.021

coordinate_white_noise = 1e-3 * np.random.normal(mu_cords, st_d_cords, hpop_coordinates.shape)
velocity_white_noise = 1e-3 * np.random.normal(mu_velocity, st_d_velocity, hpop_coordinates.shape)

sim_cords = hpop_coordinates + coordinate_white_noise
sim_velocities = hpop_velocities + velocity_white_noise


def time_to_timestamp(time):
    splitted_time = time.split(" ")
    day = splitted_time[0]
    if len(day) < 2:
        day = '0' + day
    time_stamp = "2023-04-" + day + "T" + splitted_time[-1][:8]
    return time_stamp


def convert_to_utc_timestamp(date_string):
    # Define the input format of the date string
    input_format = "%d %b %Y %H:%M:%S.%f"

    # Create a datetime object from the date string
    dt = datetime.strptime(date_string, input_format)

    # Set the timezone to UTC
    utc_tz = pytz.timezone('UTC')
    dt = dt.replace(tzinfo=utc_tz)

    # Convert the datetime object to a UTC timestamp
    timestamp = dt.timestamp()

    return timestamp


def calculate_error(predicted_position, gps_position):
    return np.linalg.norm(predicted_position - gps_position)


def calculate_checksum(line):
    """
    Calculate the checksum for a given TLE line.
    """
    checksum = 0
    for char in line[:-1]:
        if char.isdigit():
            checksum += int(char)
        elif char == "-":
            checksum += 1
    return str(checksum % 10)


def update_tle(line1, line2, inclination, raan, eccentricity, arg_perigee, mean_anomaly, mean_motion, epoch):
    # Split the TLE lines into individual components
    tle_line1 = list(line1)
    tle_line2 = list(line2)

    inclination_deg = math.degrees(inclination)
    raan_deg = math.degrees(raan)
    arg_perigee_deg = math.degrees(arg_perigee)
    mean_anomaly_deg = math.degrees(mean_anomaly)

    # Update the TLE elements with the keplerian elements
    tle_line1[18:32] = epoch  # Update the epoch in YYDDD.FFFFFFF format
    tle_line2[8:16] = '0' + f"{inclination_deg:07.4f}"  # Update the inclination in degrees
    tle_line2[17:25] = f"{raan_deg:.4f}".rjust(8)  # Update the RAAN in degrees
    tle_line2[26:34] = f"{eccentricity * 1e7:.0f}".rjust(7, '0').ljust(8)  # Update the eccentricity
    tle_line2[34:43] = f"{arg_perigee_deg:.4f}".rjust(8) + ' '  # Update the argument of perigee in degrees
    tle_line2[43:52] = f"{mean_anomaly_deg:.4f}".rjust(8) + ' '  # Update the mean anomaly in degrees
    tle_line2[52:63] = f"{mean_motion:.8f}".rjust(10)  # Update the mean motion
    # Join the updated TLE components back into strings
    tle_line1 = ''.join(tle_line1)
    tle_line2 = ''.join(tle_line2)
    return tle_line1, tle_line2


def jd_fr_to_tle_epoch(jd, fr):
    # Convert jd.fr to astropy Time object
    t = Time(jd + fr, format='jd')

    # Extract year, day of the year, and fractional time
    year = t.datetime.year % 100
    day_of_year = t.datetime.timetuple().tm_yday
    fractional_time = t.datetime.hour / 24.0 + t.datetime.minute / 1440.0 + t.datetime.second / 86400.0

    # Format DDD and fractional time strings
    ddd = '{:03d}'.format(day_of_year)
    fffffff = '{:.8f}'.format(fractional_time)[2:]

    # Construct the YYDDD.FFFFFFF string
    tle_epoch = '{:02d}{}.{:8s}'.format(year, ddd, fffffff)

    return tle_epoch


# def calculate_keplerian_elements(r, r_dot):
#     mu = 398600.435507
#     # Calculate specific angular momentum
#     h = np.cross(r, r_dot)
#
#     # Calculate eccentricity vector
#     e_vec = np.cross(r_dot, h) / mu - r / np.linalg.norm(r)
#
#     n = np.cross(np.array([0, 0, 1]), h)
#
#     if np.dot(r, r_dot) >= 0:
#         true_anomaly = np.arccos(np.dot(e_vec, r) / (np.linalg.norm(e_vec) * np.linalg.norm(r)))
#     else:
#         true_anomaly = 2 * np.pi - np.arccos(np.dot(e_vec, r) / (np.linalg.norm(e_vec) * np.linalg.norm(r)))
#
#     inclination = np.arccos(h[2] / np.linalg.norm(h))
#
#     eccentricity = np.linalg.norm(e_vec)
#     eccentric_anomaly = 2 * np.arctan(np.tan(true_anomaly / 2) /
#                                       np.sqrt((1 + eccentricity) / (1 - eccentricity)))
#
#     ascending_node = np.arctan2(n[1], n[0])
#     ascending_node = ascending_node if ascending_node >= 0 else ascending_node + 2 * np.pi
#
#     if e_vec[2] >= 0:
#         argument_of_perigee = np.arccos(np.dot(n, e_vec) / (np.linalg.norm(n) * np.linalg.norm(e_vec)))
#     else:
#         argument_of_perigee = 2 * np.pi - np.arccos(np.dot(n, e_vec) / (np.linalg.norm(n) * np.linalg.norm(e_vec)))
#
#     mean_anomaly = eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly)
#     mean_anomaly = mean_anomaly if mean_anomaly >= 0 else mean_anomaly + 2 * np.pi
#
#     r_norm = np.linalg.norm(r)  # the magnitude (norm) of the position vector
#     r_dot_norm = np.linalg.norm(r_dot)  # the magnitude (norm) of the velocity vector
#     sma = 1 / (2 / r_norm - r_dot_norm ** 2 / mu)
#
#     mean_motion = np.sqrt(mu / (sma ** 3)) * (24 * 60 * 60) / (2 * np.pi)
#     return inclination, ascending_node, eccentricity, argument_of_perigee, mean_anomaly, mean_motion


def motion_model(state, dt):
    """
    State transition function.

    Arguments:
    x -- state vector, expected order: [x_pos, x_vel, y_pos, y_vel, z_pos, z_vel]
    dt -- time step

    Returns:
    new_x -- new state vector after time step dt
    """

    # gravitational constant times the mass of the Earth, in km^3/s^2
    GM = 398600.435507

    # current position and velocity
    pos = np.array([state[0], state[1], state[2]])
    vel = np.array([state[3], state[4], state[5]])

    # calculate the gravitational acceleration
    r = np.linalg.norm(pos)
    acc = -GM / r ** 3 * pos

    # predict the new position and velocity
    new_pos = pos + vel * dt + 0.5 * acc * dt ** 2
    new_vel = vel + acc * dt

    new_state = np.zeros_like(state)
    new_state[0] = new_pos[0]
    new_state[1] = new_pos[1]
    new_state[2] = new_pos[2]
    new_state[3] = new_vel[0]
    new_state[4] = new_vel[1]
    new_state[5] = new_vel[2]

    return new_state


# Define the measurement model function
def measurement_model(state):
    return np.array([state_vars for state_vars in state])  # Return the full state vector


# Define the Jacobian matrix of the measurement model
def measurement_jacobian(state):
    return np.eye(6)  # Return an identity matrix since measurement model is linear



def get_state_from_ukf(observations, initial_guess_pos, initial_guess_v, time_step):
    position_covariance = np.diag([0.447 * 1e-3, 0.447 * 1e-3, 0.447 * 1e-3])
    velocity_covariance = np.diag([0.063 * 1e-3, 0.063 * 1e-3, 0.063 * 1e-3])

    points = MerweScaledSigmaPoints(6, alpha=0.1, beta=2., kappa=3)
    kf = UnscentedKalmanFilter(dim_x=6, dim_z=6, dt=time_step, fx=motion_model, hx=measurement_model, points=points)
    kf.x = np.hstack((initial_guess_pos, initial_guess_v))
    # initialize the state covariance
    kf.P = np.block([[position_covariance, np.zeros((3, 3))],
                     [np.zeros((3, 3)), velocity_covariance]])
    # create the process noise matrix
    kf.Q = np.eye(6) * 0.01

    # create the measurement noise matrix
    kf.R = np.eye(6) * 0.1
    for observation in observations:
        kf.predict()
        kf.update(observation)
    kf.predict()
    return kf.x


def calculate_keplerian_elements(r, v):
    mu = 398600.4418
    # Calculate specific angular momentum
    h = np.cross(r, v)

    # Calculate eccentricity vector
    e_vec = ((np.linalg.norm(v) ** 2 - mu / np.linalg.norm(r)) * r - np.dot(r, v) * v) / mu

    # Calculate eccentricity
    eccentricity = np.linalg.norm(e_vec)

    # Calculate semimajor axis
    a = 1 / (2 / np.linalg.norm(r) - np.linalg.norm(v) ** 2 / mu)

    # Calculate inclination
    inclination = np.arccos(h[2] / np.linalg.norm(h))

    # Calculate RAAN
    n = np.cross([0, 0, 1], h)
    n_norm = np.linalg.norm(n)
    if n_norm != 0:
        raan = np.arccos(n[0] / n_norm)
        if n[1] < 0:
            raan = 2 * np.pi - raan
    else:
        raan = 0

    # Calculate argument of perigee
    if eccentricity != 0:
        arg_perigee = np.arccos(np.dot(n, e_vec) / (n_norm * eccentricity))
        if e_vec[2] < 0:
            arg_perigee = 2 * np.pi - arg_perigee
    else:
        arg_perigee = 0

    # Calculate mean anomaly
    E = np.arccos((1 - np.linalg.norm(r) / a) / eccentricity)
    if np.dot(r, v) < 0:
        E = 2 * np.pi - E
    mean_anomaly = E - eccentricity * np.sin(E)

    # Calculate mean motion
    mean_motion = np.sqrt(mu / a ** 3) * (24 * 60 * 60) / (2 * np.pi)  # revs/day

    return inclination, raan, eccentricity, arg_perigee, mean_anomaly, mean_motion