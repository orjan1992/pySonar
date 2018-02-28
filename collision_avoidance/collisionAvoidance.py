class CollisionAvoidance:

    def __init__(self, msg_client):
        self.lat = self.long = self.psi = 0.0
        self.waypoint_counter = 0
        self.waypoint_list = []
        self.msg_client = msg_client
        # self.msg_client.send_msg('waypoints', str(InitialWaypoints.waypoints))

    def update_pos(self, lat=None, long=None, psi=None):
        if lat is not None:
            self.lat = lat
        if long is not None:
            self.long = long
        if psi is not None:
            self.psi = psi

    def callback(self, waypoints_list, waypoint_counter):
        self.waypoint_counter = waypoint_counter
        self.waypoint_list = waypoints_list
        # print('Counter: {}\nWaypoints: {}\n'.format(self.waypoint_counter, str(self.waypoint_list)))
        # TODO: Calculate new waypoints and send them back


if __name__ == '__main__':
    import numpy as np
    import cv2
    from settings import FeatureExtraction
    from collision_avoidance.voronoi import Voronoi

    ###################
    ### find countours
    ###################

    # Read image
    im = np.load('collision_avoidance/test.npz')['olog'].astype(np.uint8)
    # Finding histogram, calculating gradient
    hist = np.histogram(im.ravel(), 256)[0][1:]
    grad = np.gradient(hist)
    i = np.argmax(np.abs(grad) < 10)

    # threshold based on gradient
    thresh = cv2.threshold(im, i, 255, cv2.THRESH_BINARY)[1]
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Removing small contours
    new_contours = list()
    for contour in contours:
        if cv2.contourArea(contour) > FeatureExtraction.min_area:
            new_contours.append(contour)
    im2 = cv2.drawContours(np.zeros(np.shape(im), dtype=np.uint8), new_contours, -1, (255, 255, 255), 1)

    # dilating to join close contours
    im3 = cv2.dilate(im2, FeatureExtraction.kernel, iterations=FeatureExtraction.iterations)
    _, contours, _ = cv2.findContours(im3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    new_im = np.zeros((np.shape(im)[0], np.shape(im)[1], 3), dtype=np.uint8)
    new_im = cv2.drawContours(new_im, contours, -1, (255, 255, 255), -1)
    #################
    ### Prepare Voronoi
    #################

    points = []
    for contour in contours:
        for i in range(np.shape(contour)[0]):
            points.append((contour[i, 0][0], contour[i, 0][1]))


    # add start and end wp
    WP0 = (800, 801)
    WP_end = (1300, 0)
    points.append(WP0)
    points.append(WP_end)

    import scipy.spatial as sp
    from collision_avoidance.voronoi import MyVoronoi
    import matplotlib.pyplot as plt

    vp = MyVoronoi(np.array(points))
    vp.add_wp(-1)
    vp.add_wp(-2)



    # draw vertices
    for ridge in vp.ridge_vertices:
        if ridge[0] > -1 and ridge[1] > -1:
            p1x = int(vp.vertices[ridge[0]][0])
            p1y = int(vp.vertices[ridge[0]][1])
            p2x = int(vp.vertices[ridge[1]][0])
            p2y = int(vp.vertices[ridge[1]][1])
            if p1x >= 0 and p2x >= 0 and p1y >= 0 and p2y >= 0:
                cv2.line(new_im, (p1x, p1y), (p2x, p2y), (0, 255, 0), 1)

    # draw WP0 and WP_end
    cv2.circle(new_im, WP0, 2, (0, 0, 255), 2)
    cv2.circle(new_im, WP_end, 2, (0, 0, 255), 2)

    # cv2.imshow('test', new_im)
    # cv2.waitKey()

    # ################
    # ### Post-process
    # ################
    bin = cv2.drawContours(np.zeros((np.shape(im)[0], np.shape(im)[1]), dtype=np.uint8), contours, -1, (255, 255, 255), -1)
    blank_im = np.zeros(np.shape(im), dtype=np.uint8)
    im = cv2.drawContours(np.zeros(np.shape(im), dtype=np.uint8), contours, -1, (255, 255, 255), -1)
    for ridge in vp.ridge_vertices:
        if ridge[0] > -1 and ridge[1] > -1:
            p1x = int(vp.vertices[ridge[0]][0])
            p1y = int(vp.vertices[ridge[0]][1])
            p2x = int(vp.vertices[ridge[1]][0])
            p2y = int(vp.vertices[ridge[1]][1])
            if p1x >= 0 and p2x >= 0 and p1y >= 0 and p2y >= 0:
                lin = cv2.line(np.zeros(np.shape(bin), dtype=np.uint8), (p1x, p1y), (p2x, p2y), (255, 255, 255), 1)
                if not np.any(np.logical_and(bin, lin)):
                    cv2.line(new_im, (p1x, p1y), (p2x, p2y), (0, 0, 255), 1)
                    cv2.line(blank_im, (p1x, p1y), (p2x, p2y), (255, 255, 255), 1)


    # cv2.circle(im, WP0, 2, (0, 0, 255), 2)
    # cv2.circle(im, WP_end, 2, (0, 0, 255), 2)
    # cv2.imshow('test', new_im)
    # cv2.waitKey()
    cv2.imshow('test', blank_im)
    cv2.waitKey()