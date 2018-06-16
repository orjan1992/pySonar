import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d


tmp = np.load('data.npz')
contours = tmp['contours_convex']

points = []
for contour in contours:
    p1 = contour[-1, :][0]
    for i in range(np.shape(contour)[0]):
        p2 = contour[i, :][0]
        l = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
        if l > 50:
            new_x = np.linspace(p1[0], p2[0], l // 50, True)
            new_y = np.interp(new_x, [p1[0], p2[0]], [p1[1], p2[1]])
            for i in range(1, len(new_x)):
                points.append((new_x[i], new_y[i]))
        else:
            points.append((p2[0], p2[1]))
        p1 = p2

vor = Voronoi(points, False)
voronoi_plot_2d(vor)
plt.gca().invert_yaxis()
plt.show()