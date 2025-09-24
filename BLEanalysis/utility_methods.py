def normaliseXY(coordsX, coordsY):
    Xavg = np.sum(coordsX)/len(coordsX)
    Yavg = np.sum(coordsY)/len(coordsY)
    for coord in range(len(coordsX)):
        coordsX[coord] = coordsX[coord] - Xavg
    for coord in range(len(coordsY)):
        coordsY[coord] = coordsY[coord] - Yavg
    return coordsX, coordsY, Xavg, Yavg

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