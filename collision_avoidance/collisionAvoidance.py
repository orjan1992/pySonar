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
    im = cv2.applyColorMap(im, cv2.COLORMAP_HOT)
    ellipses = list()
    # contours = [np.array([[[800, 800]], [[700, 600]], [[900, 600]], [[800, 400]], [[900, 400]]])]
    for contour in contours:
        if len(contour) > 4:
            ellipse = cv2.fitEllipse(contour)
            im = cv2.ellipse(im, ellipse, (255, 0, 0), 2)
            ellipses.append(ellipse)
        else:
            rect = cv2.minAreaRect(contour)
            box = np.int32(cv2.boxPoints(rect))
            im = cv2.drawContours(im, [box], 0, (255, 0, 0), 2)
            # ellipses.append(rect)

    import time
    start = time.time()
    #################
    ### Prepare Voronoi
    #################

    points = []
    vor = np.zeros(np.shape(im), dtype=np.uint8)
    for contour in contours:
        for i in range(np.shape(contour)[0]):
            points.append((contour[i, 0][0], contour[i, 0][1]))
            # draw points
            cv2.circle(vor, (contour[i, 0][0], contour[i, 0][1]), 1, (0, 0, 255), 2)

    # add start and end wp
    WP0 = (800, 801)
    WP_end = (1300, 0)
    points.append(WP0)
    points.append(WP_end)
    vp = Voronoi(points)
    vp.process()
    print(time.time() - start)
    # lines = vp.get_output()
    # for l in vp.output:
    #     if l.end is not None:
    #         cv2.line(vor, l.start.get_point(), l.end.get_point(), (0, 255, 0), 1)

    ################
    ### Post-process
    ################
    bin = cv2.threshold(cv2.cvtColor(cv2.drawContours(np.zeros(np.shape(im), dtype=np.uint8), contours, -1, (255, 255, 255), -1), cv2.COLOR_BGR2GRAY), i, 1, cv2.THRESH_BINARY)[1]
    im = cv2.drawContours(np.zeros(np.shape(im), dtype=np.uint8), contours, -1, (255, 255, 255), -1)
    counter = 0
    lines = []
    for l in vp.output:
        if l.end is not None and l.valid():
            lin = np.zeros(np.shape(bin), dtype=np.uint8)
            lin = cv2.line(lin, l.start.get_point(), l.end.get_point(), (255, 255, 255), 1)
            if np.any(np.logical_and(bin, lin)):
                # print(np.sum(np.logical_and(bin, lin)))
                counter += 1
            else:
                cv2.line(im, l.start.get_point(), l.end.get_point(), (0, 255, 0), 1)
                lines.append(l)

    cv2.circle(im, WP0, 2, (0, 0, 255), 2)
    cv2.circle(im, WP_end, 2, (0, 0, 255), 2)
    cv2.imshow('test', im)
    cv2.waitKey()