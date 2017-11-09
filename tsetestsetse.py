from math import pi, tan
import numpy as np
import matplotlib.pyplot as plt
L = 1
# psi = np.linspace(0, pi/2, 20)
psi = 60*pi/180
y = .5*L*(1-np.tan(psi/2))
A = .25*L**2*(1-2*y)*y/(1-y)
print(A)
L = 1
y = .5*L*(1-np.tan(psi/2))
x = 0.25*L*(L/(y-L)+2)
A = x*y
print(A)



# plt.plot(psi, 4*A)
# plt.hold(True)
# plt.plot([psi[0], psi[-1]], [L**2, L**2])
# plt.show()
