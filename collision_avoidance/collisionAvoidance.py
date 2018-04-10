import numpy as np
from collision_avoidance.voronoi import MyVoronoi
from collision_avoidance.fermat import fermat
from coordinate_transformations import *
from settings import GridSettings, CollisionSettings, Settings
from time import sleep
import logging
import cv2
from time import time, strftime

logger = logging.getLogger('Collision_avoidance')
# console = logging.StreamHandler()
# console.setLevel(logging.DEBUG)
# logger.addHandler(console)

class CollisionAvoidance:
    # TODO: Should be redesigned to keep waypoint list WP = [lat, long, alt/depth?], surge speed
    save_counter = 0
    pos_and_obs_lock = False

    def __init__(self, msg_client=None, voronoi_plot_item=None):
        self.lat = self.long = self.psi = 0.0
        self.range = 0.0
        self.obstacles = []
        self.waypoint_counter = 0
        self.waypoint_list = []
        self.voronoi_wp_list = []
        self.new_wp_list = []
        if Settings.save_paths:
            self.pos = []
            self.paths = []
        self.bin_map = np.zeros((GridSettings.height, GridSettings.width), dtype=np.uint8)
        self.msg_client = msg_client
        self.voronoi_plot_item = voronoi_plot_item
        # self.msg_client.send_msg('waypoints', str(InitialWaypoints.waypoints))

    def update_pos(self, lat=None, long=None, psi=None):
        if not self.pos_and_obs_lock:
            if lat is not None:
                self.lat = lat
            if long is not None:
                self.long = long
            if psi is not None:
                self.psi = psi
        if Settings.save_paths:
            self.pos.append((lat, long, psi))

    def update_obstacles(self, obstacles, range):
        if not self.pos_and_obs_lock:
            self.obstacles = obstacles
            self.range = range

    def update_external_wps(self, wp_list, wp_counter):
        if not self.pos_and_obs_lock:
            if wp_list is not None:
                self.waypoint_list = wp_list
            if wp_counter is not None:
                self.waypoint_counter = wp_counter
        if Settings.save_paths:
            self.paths.append(wp_list)

    def main_loop(self, reliable):
        """
        Main collision avoidance loop
        :param reliable: is the occupancy grid reliable
        :return: 0 if no collision danger, 1 if new wps, 2 if collision danger but no feasible path
        """
        if reliable:
            self.pos_and_obs_lock = True
            t0 = time()
            status = 1
            if self.check_collision_margins():
                if not self.calc_new_wp():
                    logger.info('Collision danger: could not calculate feasible path')
                    status = 2
                logger.debug('collision loop time: {}'.format(time()-t0))
            else:
                logger.debug('no collision danger')
                status = 0
            self.pos_and_obs_lock = False
            return status

    def remove_obsolete_wp(self, wp_list):
        i = 0
        counter = 0
        while i < len(wp_list) - 2:
            lin = cv2.line(np.zeros(np.shape(self.bin_map), dtype=np.uint8), wp_list[i], wp_list[i+2], (255, 255, 255), 1)
            if np.any(np.logical_and(self.bin_map, lin)):
                i += 1
            else:
                wp_list.remove(wp_list[i+1])
                counter += 1
        logger.debug('{} redundant wps removed'.format(counter))
        return wp_list

    def check_collision_margins(self):
        if len(self.obstacles) > 0 and np.shape(self.waypoint_list)[0] > 0:
            self.bin_map = cv2.drawContours(np.zeros((GridSettings.height, GridSettings.width),
                                                dtype=np.uint8), self.obstacles, -1, (255, 255, 255), -1)
            k_size = np.round(CollisionSettings.obstacle_margin * 801.0 / self.range).astype(int)
            self.bin_map = cv2.dilate(self.bin_map, np.ones((k_size, k_size), dtype=np.uint8), iterations=1)

            old_wp = (self.lat, self.long)
            grid_old_wp = (801, 801)
            line_width = np.round(CollisionSettings.vehicle_margin * 801 / self.range).astype(int)
            for i in range(self.waypoint_counter, np.shape(self.waypoint_list)[0]):
                NE, constrained = constrainNED2range(self.waypoint_list[i], old_wp,
                                                     self.lat, self.long, self.psi, self.range)
                grid_wp = NED2grid(NE[0], NE[1], self.lat, self.long, self.psi, self.range)
                lin = cv2.line(np.zeros(np.shape(self.bin_map), dtype=np.uint8), grid_old_wp, grid_wp,
                               (255, 255, 255), line_width)
                old_wp = NE
                grid_old_wp = grid_wp
                if np.any(np.logical_and(self.bin_map, lin)):
                    return True
                if constrained:
                    break
            return False
    
    def calc_new_wp(self):
        # Using obstacles with collision margins for wp generation
        _, self.obstacles, _ = cv2.findContours(self.bin_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)
        if len(self.obstacles) > 0 and np.shape(self.waypoint_list)[0] > 0:
            # find waypoints in range
            last_wp = None
            constrained_wp_index = self.waypoint_counter
            for i in range(self.waypoint_counter, np.shape(self.waypoint_list)[0]):
                NE, constrained = constrainNED2range(self.waypoint_list[i], self.waypoint_list[i - 1],
                                                     self.lat, self.long, self.psi, self.range)
                if constrained:
                    constrained_wp_index = i
                    last_wp = NE
                    for i in range(constrained_wp_index, np.shape(self.waypoint_list)[0]):
                        NE, constrained = constrainNED2range(self.waypoint_list[i], self.waypoint_list[i - 1],
                                                             self.lat, self.long, self.psi, self.range)
                        if not constrained:
                            constrained_wp_index = i
                            if i < np.shape(self.waypoint_list)[0] - 1:
                                NE, constrained = constrainNED2range(self.waypoint_list[i + 1], self.waypoint_list[i],
                                                                     self.lat, self.long, self.psi, self.range)
                                if constrained:
                                    constrained_wp_index = i
                                    last_wp = NE
                    break
            if last_wp is None:
                last_wp = (self.waypoint_list[-1][0], self.waypoint_list[-1][1])
                constrained_wp_index = np.shape(self.waypoint_list)[0] - 1

            # Prepare Voronoi points
            points = []
            for contour in self.obstacles:
                for i in range(np.shape(contour)[0]):
                    points.append((contour[i, 0][0], contour[i, 0][1]))
            # add border points
            for i in range(0, GridSettings.width, CollisionSettings.border_step):
                points.append((i, 0))
                points.append((i, GridSettings.height))
            for i in range(0, GridSettings.height, CollisionSettings.border_step):
                points.append((0, i))
                points.append((GridSettings.width, i))

            use_constrain_wp = False

            if CollisionSettings.wp_as_gen_point:
                points.append(vehicle2grid(self.waypoint_list[self.waypoint_counter][3], 0, self.range))
                points.append((800, 800))
                points.append(NED2grid(last_wp[0], last_wp[1], self.lat, self.long, self.psi, self.range))
                vp = MyVoronoi(points)
                constrain_wp, _ = vp.add_wp_as_gen_point(-3)
                start_wp, start_ridges = vp.add_wp_as_gen_point(-2)
                end_wp, _ = vp.add_wp_as_gen_point(-1)
            else:
                vp = MyVoronoi(points)
                start_wp, start_ridges = vp.add_wp((801, 801))
                end_wp, _ = vp.add_wp(NED2grid(last_wp[0], last_wp[1], self.lat, self.long, self.psi, self.range))
                # TODO: Smarter calc of wp, has to be function of range and speed, also path angle?
                # Check if first wp in vehicle direction is ok
                fixed_wp = vehicle2grid(CollisionSettings.first_wp_dist, 0, self.range)
                lin = cv2.line(np.zeros(np.shape(self.bin_map), dtype=np.uint8), (801, 801), fixed_wp, (255, 255, 255),
                               1)
                if not np.any(np.logical_and(self.bin_map, lin)):
                    # Fixed wp cannot be used
                    constrain_wp, _ = vp.add_wp(fixed_wp)
                    use_constrain_wp = True

                # constrain_wp, _ = vp.add_wp(vehicle2grid(self.waypoint_list[self.waypoint_counter][3]*Settings.collision_avoidance_interval*0.0015, 0, self.range))

            # vp.gen_obs_free_connections(self.obstacles, (GridSettings.height, GridSettings.width))
            # vp.add_start_penalty(start_ridges)
            if CollisionSettings.add_path_deviation_penalty:
                old_voronoi_wp_list = [(801, 801)]
                for i in range(self.waypoint_counter, np.shape(self.waypoint_list)[0]):
                    NE, constrained = constrainNED2range(self.waypoint_list[i], self.waypoint_list[i - 1],
                                                         self.lat, self.long, self.psi, self.range)
                    old_voronoi_wp_list.append(NED2grid(NE[0], NE[1], self.lat, self.long, self.psi, self.range))
                    if constrained:
                        break
                vp.gen_obs_free_connections(self.obstacles, (GridSettings.height, GridSettings.width), True, old_voronoi_wp_list)
            else:
                vp.gen_obs_free_connections(self.obstacles, (GridSettings.height, GridSettings.width))

            self.new_wp_list = []  # self.waypoint_list[:self.waypoint_counter]
            self.voronoi_wp_list = []

            # Find shortest route
            if use_constrain_wp:
                wps = vp.dijkstra(constrain_wp, end_wp)
            else:
                wps = vp.dijkstra(start_wp, end_wp)
            if wps is not None:
                for wp in wps:
                    self.voronoi_wp_list.append((int(vp.vertices[wp][0]), int(vp.vertices[wp][1])))
                self.voronoi_wp_list = self.remove_obsolete_wp(self.voronoi_wp_list)
                self.voronoi_wp_list.insert(0, (int(vp.vertices[start_wp][0]), int(vp.vertices[start_wp][1])))
                for wps in self.voronoi_wp_list:
                    N, E = grid2NED(wps[0], wps[1], self.range, self.lat, self.long, self.psi)
                    self.new_wp_list.append([N, E, self.waypoint_list[self.waypoint_counter][2], self.waypoint_list[self.waypoint_counter][3]])
            else:
                return False
            # Smooth waypoints
            # self.new_wp_list.append(self.waypoint_list[constrained_wp_index])
            # Add old waypoints
            try:
                # self.new_wp_list.extend(self.waypoint_list[constrained_wp_index+1:])
                self.new_wp_list.extend(self.waypoint_list[constrained_wp_index:])
            except IndexError:
                pass
            self.new_wp_list = fermat(self.new_wp_list)

            # Check if smooth path is collision free
            collision = self.check_collision_margins()
            while collision:
                # TODO: Calc new path with modified dijkstra from lekkas
                logger.debug('Smooth path violates collision margins')
                collision = False

            if CollisionSettings.send_new_wps:
                self.msg_client.send_msg('new_waypoints', str(self.new_wp_list))

            if Settings.show_voronoi_plot or Settings.save_obstacles:
                im = self.calc_voronoi_img(vp, self.new_wp_list, self.voronoi_wp_list)
                if Settings.show_voronoi_plot:
                    self.voronoi_plot_item.setImage(im)
                if Settings.save_obstacles:
                    np.savez('pySonarLog/obs_{}'.format(strftime("%Y%m%d-%H%M%S")), im=new_im)
            return True
            # return vp

    def calc_voronoi_img(self, vp, wp_list, voronoi_wp_list):
        new_im = np.zeros((GridSettings.height, GridSettings.width, 3), dtype=np.uint8)
        new_im[:, :, 0] = self.bin_map
        new_im[:, :, 1] = self.bin_map
        new_im[:, :, 2] = self.bin_map

        # # draw vertices
        for ridge in vp.ridge_vertices:
            if ridge[0] != -1 and ridge[1] != -1:
                p1x = sat2uint(vp.vertices[ridge[0]][0], GridSettings.width)
                p1y = sat2uint(vp.vertices[ridge[0]][1], GridSettings.height)
                p2x = sat2uint(vp.vertices[ridge[1]][0], GridSettings.width)
                p2y = sat2uint(vp.vertices[ridge[1]][1], GridSettings.height)
                cv2.line(new_im, (p1x, p1y), (p2x, p2y), (0, 0, 255), 1)
        for i in range(np.shape(vp.connection_matrix)[0]):
            for j in range(np.shape(vp.connection_matrix)[1]):
                if vp.connection_matrix[i, j] != 0:
                    cv2.line(new_im, (sat2uint(vp.vertices[i][0], GridSettings.width),
                                      sat2uint(vp.vertices[i][1], GridSettings.height)),
                             (sat2uint(vp.vertices[j][0], GridSettings.width)
                              , sat2uint(vp.vertices[j][1], GridSettings.height)), (0, 255, 0), 1)


        if len(voronoi_wp_list) > 0:
            for i in range(len(voronoi_wp_list) - 1):
                WP1 = voronoi_wp_list[i]
                cv2.circle(new_im, voronoi_wp_list[i], 2, (0, 0, 255), 2)
                cv2.line(new_im, voronoi_wp_list[i], voronoi_wp_list[i+1], (255, 0, 0), 2)
            cv2.circle(new_im, voronoi_wp_list[-1], 2, (0, 0, 255), 2)
        return new_im

    def save_paths(self):
        if Settings.save_paths:
            np.savez('pySonarLog/paths_{}'.format(strftime("%Y%m%d-%H%M%S")), paths=np.array(self.paths), pos=np.array(self.pos))

if __name__ == '__main__':
    import cv2
    from settings import FeatureExtraction

    ###################
    ### find countours
    ###################

    # Read image
    im = np.load('test.npz')['olog'].astype(np.uint8)
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
    # for contour in contours:
    #     for i in range(np.shape(contour)[0]):
    #         cv2.circle(new_im, (contour[i, 0][0], contour[i, 0][1]), 2, (255, 0, 0), -1)

    ######
    ## init
    ######
    CollisionSettings.send_new_wps = False
    collision_avoidance = CollisionAvoidance(None)
    collision_avoidance.update_pos(0, 0, 0)
    # collision_avoidance.waypoint_list = [[0, 0, 1, 1], [10, 10, 2, 2], [15, 20, 3, 3]]
    # collision_avoidance.waypoint_list = [[0, 0, 1, 1], [15, 20, 3, 3]]
    # collision_avoidance.waypoint_list = [[0, 0, 1, 1], [29, -13, 3, 3]]
    collision_avoidance.waypoint_list = [[0, 0, 1, 1], [59, 10, 3, 3]]
    collision_avoidance.waypoint_counter = 1
    collision_avoidance.update_obstacles(contours, 30)

    #################
    ### Check for collision
    #################

    # im = collision_avoidance.check_collision_margins()
    # im = cv2.drawContours(im, contours, -1, (0, 0, 255), -1)
    # cv2.imshow('sdf', im)
    # cv2.waitKey()
    #################
    ### Prepare Voronoi
    #################
    collision_avoidance.check_collision_margins()
    vp = collision_avoidance.calc_new_wp()
    wp_list = collision_avoidance.new_wp_list
    voronoi_wp_list = collision_avoidance.voronoi_wp_list

    new_im = np.zeros((GridSettings.height, GridSettings.width, 3), dtype=np.uint8)
    new_im[:, :, 0] = collision_avoidance.bin_map
    new_im[:, :, 1] = collision_avoidance.bin_map
    new_im[:, :, 2] = collision_avoidance.bin_map

    # # draw vertices
    for ridge in vp.ridge_vertices:
        if ridge[0] != -1 and ridge[1] != -1:
            p1x = sat2uint(vp.vertices[ridge[0]][0], GridSettings.width)
            p1y = sat2uint(vp.vertices[ridge[0]][1], GridSettings.height)
            p2x = sat2uint(vp.vertices[ridge[1]][0], GridSettings.width)
            p2y = sat2uint(vp.vertices[ridge[1]][1], GridSettings.height)
            cv2.line(new_im, (p1x, p1y), (p2x, p2y), (0, 0, 255), 1)
    for i in range(np.shape(vp.connection_matrix)[0]):
        for j in range(np.shape(vp.connection_matrix)[1]):
            if vp.connection_matrix[i, j] != 0:
                cv2.line(new_im, (sat2uint(vp.vertices[i][0], GridSettings.width),
                                  sat2uint(vp.vertices[i][1], GridSettings.height)),
                         (sat2uint(vp.vertices[j][0], GridSettings.width)
                          , sat2uint(vp.vertices[j][1], GridSettings.height)), (0, 255, 0), 1)

    # draw route
    WP0 = NED2grid(collision_avoidance.new_wp_list[0][0], collision_avoidance.new_wp_list[0][1], 0, 0, 0, 30)
    cv2.circle(new_im, WP0, 2, (0, 0, 255), 2)
    for i in range(len(collision_avoidance.new_wp_list)-1):
        NE, constrained = constrainNED2range(collision_avoidance.new_wp_list[i+1], collision_avoidance.new_wp_list[i], 0, 0, 0, 30)
        if not constrained:
            WP1 = NED2grid(collision_avoidance.new_wp_list[i+1][0], collision_avoidance.new_wp_list[i+1][1], 0, 0, 0, 30)
            cv2.circle(new_im, WP1, 2, (255, 0, 0), 2)
            cv2.line(new_im, WP0, WP1, (255, 0, 0), 2)
            WP0 = WP1
        else:
            WP1 = NED2grid(NE[0], NE[1], 0, 0,
                           0, 30)
            cv2.circle(new_im, WP1, 2, (255, 0, 0), 2)
            cv2.line(new_im, WP0, WP1, (255, 0, 0), 2)
            break

    cv2.rectangle(new_im, (0, 0), (GridSettings.width-1, GridSettings.height), (255, 255, 255), 1)
    cv2.imshow('test', new_im)
    cv2.waitKey()