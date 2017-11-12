import numpy as np
from messages.posMsg import PosMsg
from math import pi, sin, cos

m1 = PosMsg()
m1.lat = 1
m1.long = 1
m1.head = pi/4

m2 = PosMsg()
m2.lat = 0
m2.long = 0
m2.head = 0

msg = PosMsg()
R = np.array([[cos(m1.head), -sin(m1.head)], [sin(m1.head), cos(m1.head)]])
lat_long = np.array([[(m1.lat - m2.lat)], [(m1.long - m2.long)]])
xy = np.dot(R.T, lat_long)
msg.x = xy[0]
msg.y = xy[1]
msg.head = m1.head - m2.head

print('Transformed grid: delta_x: {}\tdelta_y: {}\tdelta_psi: {} deg'.format(msg.x, msg.y, msg.head*180/pi))
print(lat_long)
print(np.shape(lat_long))