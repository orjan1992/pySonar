import numpy as np
import math
from pathlib import Path

class OGrid(object):
    OZero = 0.5
    cur_step = 0
    GRAD2RAD = math.pi/(16 * 200)
    RAD2GRAD = (16*200)/math.pi
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
                self.origoJ = round(self.X / 2)
                self.origoI = self.Y
                self.oLog = np.ones((self.Y, self.X)) * self.OZero
                self.O_logic = np.zeros((self.Y, self.X), dtype=bool)
                [self.iMax, self.jMax] = np.shape(self.oLog)

                fStr = 'OGrid_data/angleRad_X=%i_Y=%i_size=%i.npz' % (self.X, self.Y, int(cellSize*100))
                try:
                    tmp = np.load(fStr)
                    self.r = tmp['r']
                    self.rHigh = tmp['rHigh']
                    self.rLow = tmp['rLow']
                    self.theta = tmp['theta']
                    self.thetaHigh = tmp['thetaHigh']
                    self.thetaLow = tmp['thetaLow']
                except FileNotFoundError:
                    # Calculate angles and radii
                    self.r = np.zeros((self.Y, self.X))
                    self.rHigh = np.zeros((self.Y, self.X))
                    self.rLow = np.zeros((self.Y, self.X))
                    self.theta = np.zeros((self.Y, self.X))
                    self.thetaHigh = np.zeros((self.Y, self.X))
                    self.thetaLow = np.zeros((self.Y, self.X))
                    for i in range(0, self.Y):
                        for j in range(0, self.X):
                            x = (j - self.origoJ) * self.cellSize
                            y = (self.origoI - i) * self.cellSize
                            self.r[i, j] = math.sqrt(pow(x, 2) + pow(y, 2))
                            print(math.sqrt(x**2 + y**2))
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
                    print(self.r)
                    np.savez(fStr, r=self.r, rHigh=self.rHigh, rLow=self.rLow, theta=self.theta, thetaHigh=self.thetaHigh, thetaLow=self.thetaLow)
            self.steps = np.array([4, 8, 16, 32])
            self.bearing_ref = np.linspace(-math.pi/2, math.pi/2, self.RAD2GRAD)
            self.mappingMax = 2*self.X;
            self.makeMap(self.steps)
            self.loadMap(self.steps[0]*self.GRAD2RAD)

    def makeMap(self, step_angle_size):

        filename_base  = 'OGrid_data/Step_X=%i_Y=%i_size=%i_step=' % (self.X, self.Y, int(self.cellSize*100))
        steps_to_create = []
        for i in range(0, step_angle_size.shape[0]):
            if not Path('%s%i.npz'%(filename_base, step_angle_size[i])).is_file():
                steps_to_create.append(step_angle_size[i])
        
        for steps_to in steps_to_create:
            #Create  Mapping
            step = np.array(steps_to_create)*self.GRAD2RAD
            for step_i in step:
                mapping = np.zeros((len(self.bearing_ref), self.mappingMax), dtype=np.dtype('u4'))
                for i in range(0, len(self.bearing_ref)):
                    cells = self.sonarCone(step_i, self.bearing_ref[i])
                    try:
                        mapping[i, 0:len(cells)] = cells
                    except ValueError as error:
                        print('Mapping variable to small !!!! %s' % error)
                # Saving to file
                np.savez('%s%i.npz'%(filename_base, steps_to), mapping=mapping)

    def loadMap(self, step):
        # LOADMAP Loads the map. Step is in rad
        step = round(step*self.RAD2GRAD)
        if self.cur_step != step or not self.mapping:
            if not any(np.nonzero(self.steps == step)):
                self.makeMap(np.array([step]))
            try:
                self.mapping = np.load('OGrid_data/Step_X=%i_Y=%i_size=%i_step=%i.npz' % (self.X, self.Y, int(self.cellSize * 100), step))['mapping']
            except FileNotFoundError:
                print('Could not find mapping file!')
            self.cur_step = step

    def sonarCone(self, step, theta):
        theta1 = max(theta - step, -math.pi/2)
        theta2 = min(theta + step, math.pi/2)
        (row, col) = np.nonzero(self.thetaLow <= theta2)
        a = self.sub2ind(row, col)
    
        (row, col) = np.nonzero(self.thetaHigh >= theta1)
        b = self.sub2ind(row, col)
        return np.intersect1d(a, b)

    def sub2ind(self, row, col):
        return row + col*self.iMax

    def sonarConeLookup(self, step, theta):
        self.loadMap(step)
        [tmp, j] = min(abs(theta - self.bearing_ref))
        cone = self.mapping[j, -1]
        return cone(cone != 0)