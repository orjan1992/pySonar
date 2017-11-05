import numpy as np
import math
from pathlib import Path

class OGrid(object):
    OZero = 0.5
    def __init__(self, cellSize, sizeX, sizeY, p_m):
        if cellSize > 0:
            if (sizeX > cellSize) or (sizeY > cellSize):
                if round(sizeX / cellSize)%2 == 0:
                    sizeX = sizeX + cellSize
                    print('Extended grid by one cell in X direction to make it even')


                self.XLimMeters = sizeX / 2
                self.YLimMeters = sizeY
                self.cellSize = cellSize
                self.X = round(sizeX / cellSize)
                self.Y = round(sizeY / cellSize)
                self.origoJ = round(X / 2)
                self.origoI = Y
                self.oLog = np.ones((Y, X)) * self.OZero
                self.O_logic = np.zeros((Y, X), dtype=bool)
                [self.iMax, self.jMax] = np.shape(self.oLog)
                self.steps = np.array([4, 8, 16, 32])

                fStr = 'OGrid_data/angleRad_X=%i_Y=%i_size=%i.npz' % (self.X, self.Y, int(cellSize*100))
                try:
                    tmp = np.load(fStr)
                    print(tmp.files)
                    self.r = tmp['r']
                    self.rHigh = tmp['rHigh']
                    self.rLow = tmp['rLow']
                    self.theta = tmp['theta']
                    self.thetaHigh = tmp['thetaHigh']
                    self.thetaLow = tmp['thetaLow']
                except FileNotFoundError:
                    # Calculate angles and radii
                    self.r = np.zeros((Y, X))
                    self.rHigh = self.r
                    self.rLow = self.r
                    self.theta = self.r
                    self.thetaHigh = self.r
                    self.thetaLow = self.r
                    for i in range(0, Y):
                        for j in range(0, self.jMax):
                            x = (j - self.origoJ) * self.cellSize
                            y = (self.origoI - i) * self.cellSize
                            self.r[i, j] = math.sqrt(x**2 + y**2)
                            self.theta[i, j] = math.atan2(x, y)
                            #ranges
                            self.rHigh[i, j] = math.sqrt((x + np.sign(x) * self.cellSize / 2)**2 + (y + self.cellSize / 2)**2)
                            self.rLow[i, j] = math.sqrt((x - np.sign(x) * self.cellSize / 2)**2 + (max(y - self.cellSize/2, 0))**2)

                            #angles
                            if x < 0:
                                self.thetaLow[i, j] = math.atan2(x - self.cellSize / 2, y - self.cellSize / 2)
                                self.thetaHigh[i, j] = math.atan2(x + self.cellSize / 2, y + self.cellSize / 2)
                            elif x > 0:
                                self.thetaLow[i, j] = math.atan2(x - self.cellSize / 2, y + self.cellSize / 2)
                                self.thetaHigh[i, j] = math.atan2(x + self.cellSize / 2, y - self.cellSize / 2)
                            else:
                                self.thetaLow[i, j] = math.atan2(x - self.cellSize / 2, y - self.cellSize / 2)
                                self.thetaHigh[i, j] = math.atan2(x + self.cellSize / 2, y - self.cellSize / 2)
                    #file = open(fStr)
                    np.savez(fStr, r=self.r, rHigh=self.rHigh, rLow=self.rLow, theta=self.theta, thetaHigh=self.thetaHigh, thetaLow=self.thetaLow)

    def makeMap(self, step_angle_size):

        filename_base  = 'OGrid_data/Step_X=%i_Y=%i_size=%i_step=' % (self.X, self.Y, int(self.cellSize*100))
        steps_to_create = list()
        for i in range(0, len(step_angle_size)):
            if not Path('%s%i.npz'%(filename_base, step_angle_size[i])).is_file():
                steps_to_create = steps_to_create.append(step_angle_size[i])
        bearing = range(-math.pi/2, math.pi/2, math.pi/(16*200))
        #self.bearing_ref = bearing
        
        if steps_to_create:
            #Create  Mapping
            step = steps_to_create*math.pi/(16 * 200)
            for k in range(0, step):
                mapping = np.zeros((len(bearing), 2), dtype=np.dtype('u4'))
                for i in range(0, len(bearing)):
                    cells = self.sonarCone(step[k], bearing[i])
                    mapping[i, 0:len(cells)] = cells
                # Saving to file
                np.savez('%s%i.npz'%(filename_base, steps_to_create[k]))

    def sonarCone(self, step, theta):
        theta1 = max(theta - step, -math.pi/2)
        theta2 = min(theta + step, math.pi/2)
    
        [row, col] = find(self.thetaLow <= theta2)
        a = sub2ind(size(self.O), row, col)
    
        [row, col] = find(self.thetaHigh >= theta1)
        b = sub2ind(size(self.O), row, col)
        cone = intersect(a, b)