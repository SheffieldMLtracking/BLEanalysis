import numpy as np
import pyproj

def normaliseXY(coordsX, coordsY):
    """
    Normalises single array
    """
    Xavg = np.sum(coordsX)/len(coordsX)
    Yavg = np.sum(coordsY)/len(coordsY)
    for coord in range(len(coordsX)):
        coordsX[coord] = coordsX[coord] - Xavg
    for coord in range(len(coordsY)):
        coordsY[coord] = coordsY[coord] - Yavg
    return coordsX, coordsY, Xavg, Yavg


def take_multiple_GPS_arrays_project_into_XY_and_normalise(x_one, y_one, x_two, y_two):
    """
    Ugly approach to normalising different coordinates, could be generalised to N arrays
    """
    print(x_one)

    print(x_two)
    xs = np.concatenate((x_one, x_two))
    ys = np.concatenate((y_one, y_two))

    P = pyproj.Proj(proj='utm', zone=30, ellps='WGS84', preserve_units=True)
    proj_x, proj_y = P(xs, ys)
    coordsx, coordsy, _, _ = normaliseXY(proj_x, proj_y)

    # split back into original arrays:
    x_one_normalised = coordsx[0:(len(x_one))]
    x_two_normalised = coordsx[len(x_one):]
    y_one_normalised = coordsy[0:(len(y_one))]
    y_two_normalised = coordsy[len(y_one):]
    return x_one_normalised, y_one_normalised, x_two_normalised, y_two_normalised
    
    


def reverseNormalizeXY(coordsX, coordsY, Xavg, Yavg):
    for coord in range(len(coordsX)):
        coordsX[coord] = coordsX[coord] + Xavg
    for coord in range(len(coordsY)):
        coordsY[coord] = coordsY[coord] + Yavg
    return coordsX, coordsY

def LoadFile(fileName):
  with open(fileName, 'r') as f:
    filedata = f.read()
    return filedata
