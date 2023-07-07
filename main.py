import astropy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import colorednoise as cn

from utils import time_to_timestamp

# hpop = pd.read_fwf("HPOP_Sat TEMEofEpoch Position Velocity.txt")
hpop = np.genfromtxt("hpopepoch.txt", dtype=str,
                     encoding=None, delimiter=",").astype(float)
# hpop.columns = sgp4.columns
orbital_elements = pd.read_csv("orbital_elements_hpop.csv")

hpop_coordinates = hpop[:, 0:3]

hpop_velocities = hpop[:, 3:]

# error_coords = hpop_coordinates - sgp4_coordinates

# error_velocities = hpop_velocities - sgp4_velocities

# fig, axis = plt.subplots(3)
# axis[0].plot(error_coords[0])
# axis[1].plot(error_coords[1])
# axis[2].plot(error_coords[2])
# plt.show()




# from astropy.time import Time
# from astropy.coordinates import TEME, ITRS, EarthLocation, CartesianRepresentation
#
# time = hpop["Time (UTCG)"][0]
# time_stamp = time_to_timestamp(time)
#
# time = Time(time_stamp, scale='utc')
# (x, y, z) = hpop_coordinates[:, 0]
#
# wgs84_cords = CartesianRepresentation(x, y, z)
# wgs84_frame = ITRS(wgs84_cords, obstime=time)
#
# teme_coordinates = wgs84_frame.transform_to(TEME(obstime=time))
# print(np.array(teme_coordinates))




