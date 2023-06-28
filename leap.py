from skyfield.api import Topos
from sgp4.api import Satrec, WGS72

def cartesian_to_tle(x_cartesian, y_cartesian, z_cartesian, timestamp):
    # Specify the Cartesian coordinates in TEME frame
    teme_pos = [x_cartesian, y_cartesian, z_cartesian]  # [meters]

    # Create a Satrec object with TEME Cartesian coordinates
    satellite = Satrec()
    satellite.sgp4init(
        WGS72,           # Earth gravity wgs72 model
        'i',             # Method selection, 'a' = old AFSPC, 'i' = improved IOD
        ' ',             # Satrec classification (not used for TLE generation)
        0,               # International designator (not used for TLE generation)
        0,               # Epoch time (not used for TLE generation)
        0,               # Bstar drag term (not used for TLE generation)
        0,               # Element set type (not used for TLE generation)
        teme_pos,        # Cartesian TEME coordinates [meters]
        timestamp       # Observation time (Python datetime object)
    )

    # Generate the TLE string
    line1, line2 = satellite.sgp4tle()

    return line1, line2

# Example usage
x_cartesian = 6968.782e3  # X coordinate in meters
y_cartesian = 4601.504e3  # Y coordinate in meters
z_cartesian = 6303.564e3  # Z coordinate in meters
timestamp = '2023-05-29T12:00:00'

line1, line2 = cartesian_to_tle(x_cartesian, y_cartesian, z_cartesian, timestamp)

print("TLE Lines:")
print(line1)
print(line2)
