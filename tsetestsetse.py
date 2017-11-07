import numpy as np
from math import pi
from readLogFile.wrap2pi import Wrap2pi
a = np.array(range(0, 405, 45))*pi/180
print(Wrap2pi(a+pi)*180/pi)