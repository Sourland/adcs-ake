import numpy as np
from utils import sim_cords, sim_velocities

from scipy.integrate import solve_ivp
import numpy as np

mu = 398600.4418  # gravitational parameter


# define the system of ODEs
def eom(t, y):
    r = np.sqrt(y[0] ** 2 + y[1] ** 2 + y[2] ** 2)
    return [y[3], y[4], y[5], -mu * y[0] / r ** 3, -mu * y[1] / r ** 3, -mu * y[2] / r ** 3]


# initial conditions [x0, y0, z0, vx0, vy0, vz0]
y0 = [7000, 0, 0, 0, 7.5460491, 0]  # sample initial conditions

# solve the system of ODEs
sol = solve_ivp(eom, [0, 120], y0, method='RK45')  # solve for the next 120 minutes

# print final conditions
print(sol.y[:, -1])

gps_data = np.hstack((sim_cords, sim_velocities))
state_vector_data = gps_data[:400]

print(gps_data[-1])
