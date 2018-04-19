import numpy as np
from collision_avoidance.voronoi import MyVoronoi
from collision_avoidance.fermat import fermat
from coordinate_transformations import *
from settings import GridSettings, CollisionSettings, Settings, PlotSettings
from enum import Enum
import logging
import cv2
import threading
from time import time, strftime

if Settings.input_source == 0:
    import messages.udpMsg as udpMsg

logger = logging.getLogger('Collision_avoidance')
# console = logging.StreamHandler()
# console.setLevel(logging.DEBUG)
# logger.addHandler(console)

class CollisionAvoidance:
    # TODO: Should be redesigned to keep waypoint list WP = [lat, long, alt/depth?], surge speed
    save_counter = 0
    path_ok = True

    def __init__(self, msg_client=None, voronoi_plot_item=None):
        self.lat = self.long = self.psi = 0.0
        self.range = 0.0
        self.obstacles = []
        self.waypoint_counter = 0
        self.waypoint_list = []
        self.voronoi_wp_list = []
        self.new_wp_list = []
        self.data_storage = CollisionData()
        if Settings.save_paths:
            self.pos = []
            self.paths = []
        self.bin_map = np.zeros((GridSettings.height, GridSettings.width), dtype=np.uint8)
        self.msg_client = msg_client
        self.voronoi_plot_item = voronoi_plot_item

    def update_pos(self, lat=None, long=None, psi=None):
        self.data_storage.update_pos(lat, long, psi)
        if Settings.save_paths:
            self.pos.append((lat, long, psi))

    def update_obstacles(self, obstacles, range):
        self.data_storage.update_obstacles(obstacles, range)

    def update_external_wps(self, wp_list, wp_counter):
        self.path_ok = True
        self.data_storage.update_wps(wp_list, wp_counter)
        if Settings.save_paths:
            self.paths.append(wp_list)

    def main_loop(self, reliable):
        """
        Main collision avoidance loop
        :param reliable: is the occupancy grid reliable
        :return: 0 if no collision danger, 1 if new wps, 2 if collision danger but no feasible path
        """
        if reliable:
            t0 = time()
            self.lat, self.long, self.psi = self.data_storage.get_pos()
            self.waypoint_list, self.waypoint_counter = self.data_storage.get_wps()
            self.obstacles, self.range = self.data_storage.get_obstacles()
            self.bin_map = cv2.drawContours(np.zeros((GridSettings.height, GridSettings.width),
                                                     dtype=np.uint8), self.obstacles, -1, (255, 255, 255), -1)
            if self.check_collision_margins(self.waypoint_list)[0]:
                logger.info("Collision path detected, start new wp calc")
                stat = self.calc_new_wp()
                if stat == CollisionStatus.NO_FEASIBLE_ROUTE:
                    logger.info('Collision danger: could not calculate feasible path')
                    # TODO: Implement better handling, mean of left/right half etc
                    self.msg_client.send_autopilot_msg(udpMsg.AutoPilotTrackingSpeed(0))
                elif stat == CollisionStatus.SMOOTH_PATH_VIOLATES_MARGIN:
                    logger.info('Smooth path violates margin')
                elif stat == CollisionStatus.NEW_ROUTE_OK:
                    logger.info('New route ok. Time: {}'.format(time()-t0))
            else:
                # logger.info('No collision danger')
                stat = CollisionStatus.NO_DANGER
            return stat

    def remove_obsolete_wp(self, wp_list):
        i = 0
        counter = 0
        line_width = np.round(CollisionSettings.vehicle_margin * 801 / self.range).astype(int)
        while i < len(wp_list) - 2:
            lin = cv2.line(np.zeros(np.shape(self.bin_map), dtype=np.uint8), wp_list[i], wp_list[i+2],
                           (255, 255, 255), line_width)
            if np.any(np.logical_and(self.bin_map, lin)):
                i += 1
            else:
                wp_list.remove(wp_list[i+1])
                counter += 1
        logger.debug('{} redundant wps removed'.format(counter))
        return wp_list

    def check_collision_margins(self, wp_list):
        """
        Check if path is colliding
        :param wp_list:
        :return: True if path is going outside collision margins, index of last wp before collision
        """
        if np.shape(wp_list)[0] > 0:
            old_wp = (self.lat, self.long)
            grid_old_wp = (801, 801)
            line_width = np.round(CollisionSettings.vehicle_margin * 801 / self.range).astype(int)
            for i in range(self.waypoint_counter, np.shape(wp_list)[0]):
                NE, constrained = constrainNED2range(wp_list[i], old_wp,
                                                     self.lat, self.long, self.psi, self.range)
                grid_wp = NED2grid(NE[0], NE[1], self.lat, self.long, self.psi, self.range)
                lin = cv2.line(np.zeros(np.shape(self.bin_map), dtype=np.uint8), grid_old_wp, grid_wp,
                               (255, 255, 255), line_width)
                old_wp = NE
                grid_old_wp = grid_wp
                if np.any(np.logical_and(self.bin_map, lin)):
                    return True, i - 1
                if constrained:
                    break
        return False, 0
    
    def calc_new_wp(self):
        # Using obstacles with collision margins for wp generation
        if len(self.obstacles) > 0 and np.shape(self.waypoint_list)[0] > 0:
            # Find waypoints in grid
            last_wp = None
            constrained_wp_index = self.waypoint_counter
            for i in range(self.waypoint_counter, np.shape(self.waypoint_list)[0]):
                NE, constrained = constrainNED2range(self.waypoint_list[i], self.waypoint_list[i - 1],
                                                     self.lat, self.long, self.psi, self.range)
                if constrained:
                    constrained_wp_index = i
                    last_wp = NE
                    # Check if path reenters grid
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
            # points.extend(self.border_step_list)
            constrained_wp_grid = NED2grid(self.waypoint_list[constrained_wp_index][0],
                                           self.waypoint_list[constrained_wp_index][1],
                                           self.lat, self.long, self.psi, self.range)
            x_min = min(constrained_wp_grid[0], 0)-1
            x_max = max(constrained_wp_grid[0], GridSettings.height)+1
            y_min = min(constrained_wp_grid[1], 0)-1
            y_max = max(constrained_wp_grid[1], GridSettings.width)+1
            points.extend([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)])

            use_constraint_wp = False

            # Initializing voronoi and adding wps
            vp = MyVoronoi(points)
            start_wp, start_region = vp.add_wp((801, 801))
            end_wp, end_region = vp.add_wp(constrained_wp_grid)
            # TODO: Smarter calc of wp, has to be function of range and speed, also path angle?

            # Check if first wp in vehicle direction is ok
            fixed_wp = vehicle2grid(CollisionSettings.first_wp_dist, 0, self.range)
            lin = cv2.line(np.zeros(np.shape(self.bin_map), dtype=np.uint8), (801, 801), fixed_wp, (255, 255, 255),
                           np.round(CollisionSettings.vehicle_margin * 801 / self.range).astype(int))
            if not np.any(np.logical_and(self.bin_map, lin)):
                # Fixed wp can be used
                constraint_wp, _ = vp.add_wp(fixed_wp)
                use_constraint_wp = True

            vp.gen_obs_free_connections(self.range, self.bin_map)
            self.new_wp_list = []  # self.waypoint_list[:self.waypoint_counter]
            self.voronoi_wp_list = []

            collision_danger = True
            collision_index = 0
            counter = 0
            wps = None
            skip_smoothing = False
            while collision_danger and counter < 5:
                # Find shortest route
                if counter > 0:
                    print('smooth path violates constraints, trying again: {}'.format(counter))
                try:
                    if wps is None or len(wps) == 0:
                        if use_constraint_wp:
                            wps = vp.dijkstra(constraint_wp, end_wp, None)
                        else:
                            wps = vp.dijkstra(start_wp, end_wp, None)
                    else:
                        old_wps = wps.copy()
                        wps = wps[:collision_index]
                        if len(wps) > 1 and 0 < collision_index < len(wps):
                            wps.extend(vp.dijkstra(old_wps[collision_index-1], end_wp, old_wps[collision_index]))
                        else:
                            wps = vp.dijkstra(start_wp, end_wp, None)
                except RuntimeError:
                    if 1 < counter < 5:
                        logger.info('Relaxing smoothing beacause of infeasible route')
                        skip_smoothing = True
                        wps = old_wps.copy()
                    else:
                        self.path_ok = False
                        if Settings.show_voronoi_plot:
                            im = self.calc_voronoi_img(vp, None, start_wp, end_wp, end_region, start_region)
                            self.voronoi_plot_item.setImage(im)
                        return CollisionStatus.NO_FEASIBLE_ROUTE

                for wp in wps:
                    self.voronoi_wp_list.append((int(vp.vertices[wp][0]), int(vp.vertices[wp][1])))
                self.voronoi_wp_list = self.remove_obsolete_wp(self.voronoi_wp_list)
                self.voronoi_wp_list.insert(0, (int(vp.vertices[start_wp][0]), int(vp.vertices[start_wp][1])))
                for wp in self.voronoi_wp_list:
                    N, E = grid2NED(wp[0], wp[1], self.range, self.lat, self.long, self.psi)
                    self.new_wp_list.append([N, E, self.waypoint_list[self.waypoint_counter][2], self.waypoint_list[self.waypoint_counter][3]])

                # Smooth waypoints
                if not skip_smoothing:
                    self.new_wp_list = fermat(self.new_wp_list)

                # Check if smooth path is collision free
                collision_danger, collision_index = self.check_collision_margins(self.new_wp_list)
                counter += 1
            if collision_danger:
                logger.debug('Smooth path violates collision margins')
                self.path_ok = False
                self.data_storage.update_wps(self.new_wp_list, 0)
                return CollisionStatus.SMOOTH_PATH_VIOLATES_MARGIN

            # Add waypoints outside grid
            try:
                self.new_wp_list.extend(self.waypoint_list[constrained_wp_index+1:])
            except IndexError:
                pass

            if CollisionSettings.send_new_wps:
                if Settings.input_source == 1:
                    self.msg_client.send_msg('new_waypoints', str(self.new_wp_list))
                else:
                    self.path_ok = True
                    self.msg_client.update_wps(self.new_wp_list)
                    self.data_storage.update_wps(self.new_wp_list, 0)

            if Settings.show_voronoi_plot or Settings.save_obstacles:
                im = self.calc_voronoi_img(vp, self.voronoi_wp_list, start_wp, end_wp, end_region, start_region)
                if Settings.show_voronoi_plot:
                    self.voronoi_plot_item.setImage(im)
                if Settings.save_obstacles:
                    np.savez('pySonarLog/obs_{}'.format(strftime("%Y%m%d-%H%M%S")), im=im)
            return CollisionStatus.NEW_ROUTE_OK
            # return vp

    def calc_voronoi_img(self, vp, voronoi_wp_list, start_wp=None, end_wp=None, end_region=None, start_region=None):
        x_min = np.min(vp.points[:, 0])-1000
        x_max = np.max(vp.points[:, 0])+1000
        y_min = np.min(vp.points[:, 1])-1000
        y_max = np.max(vp.points[:, 1])+1000

        vp.vertices[:, 0] -= x_min
        vp.vertices[:, 1] -= y_min
        new_im = np.zeros((int(y_max-y_min), int(x_max-x_min), 3), dtype=np.uint8)
        new_im[-int(y_min):-int(y_min)+1601, -int(x_min):-int(x_min)+1601, 0] = self.bin_map
        new_im[-int(y_min):-int(y_min)+1601, -int(x_min):-int(x_min)+1601, 1] = self.bin_map
        new_im[-int(y_min):-int(y_min)+1601, -int(x_min):-int(x_min)+1601, 2] = self.bin_map

        # Draw grid limits
        cv2.rectangle(new_im, (-int(x_min), -int(y_min)), (-int(x_min)+1601, -int(y_min)+1601), (255, 255, 255), 3)

        line_width = np.round(CollisionSettings.vehicle_margin * 801 / self.range).astype(int)
        # # draw vertices
        for ridge in vp.ridge_vertices:
            if ridge[0] != -1 and ridge[1] != -1:
                p1x = int(vp.vertices[ridge[0]][0])
                p1y = int(vp.vertices[ridge[0]][1])
                p2x = int(vp.vertices[ridge[1]][0])
                p2y = int(vp.vertices[ridge[1]][1])
                cv2.line(new_im, (p1x, p1y), (p2x, p2y), (0, 0, 255), line_width)
        for i in range(np.shape(vp.connection_matrix)[0]):
            for j in range(np.shape(vp.connection_matrix)[1]):
                if vp.connection_matrix[i, j] != 0:
                    cv2.line(new_im, (int(vp.vertices[i][0]), int(vp.vertices[i][1])),
                             (int(vp.vertices[j][0]), int(vp.vertices[j][1])), (0, 255, 0), line_width)


        if voronoi_wp_list is not None and len(voronoi_wp_list) > 0:
            voronoi_wp_list = np.array(voronoi_wp_list)
            voronoi_wp_list[:, 0] -= int(x_min)
            voronoi_wp_list[:, 1] -= int(y_min)
            voronoi_wp_list = voronoi_wp_list.tolist()
            for i in range(len(voronoi_wp_list) - 1):
                WP1 = voronoi_wp_list[i]
                cv2.circle(new_im, (voronoi_wp_list[i][0], voronoi_wp_list[i][1]), line_width, (0, 0, 255), 2)
                cv2.line(new_im, (voronoi_wp_list[i][0], voronoi_wp_list[i][1]), (voronoi_wp_list[i+1][0], voronoi_wp_list[i+1][1]), (255, 0, 0), line_width)
            cv2.circle(new_im, (voronoi_wp_list[-1][0], voronoi_wp_list[-1][1]), line_width, (0, 0, 255), 2)

        if start_wp is not None:
            cv2.circle(new_im, (int(vp.vertices[start_wp][0]), int(vp.vertices[start_wp][1])), 15, (204, 0, 255), 6)
        if end_wp is not None:
            cv2.circle(new_im, (int(vp.vertices[end_wp][0]), int(vp.vertices[end_wp][1])), 15, (204, 0, 255), 6)


        vp.points[:, 0] -= x_min
        vp.points[:, 1] -= y_min
        for point in vp.points:
            cv2.circle(new_im, (int(point[0]), int(point[1])), 10, (255, 255, 0), 4)
        if end_region is not None:
            ind = np.array(vp.regions[end_region])
            tmp = vp.vertices[ind[ind >= 0]].astype(int)
            for i in range(np.shape(tmp)[0]):
                cv2.line(new_im, (tmp[i, 0], tmp[i, 1]), (tmp[i-1, 0], tmp[i-1, 1]), (204, 0, 255), 10)
        if start_region is not None:
            ind = np.array(vp.regions[start_region])
            tmp = vp.vertices[ind[ind >= 0]].astype(int)
            for i in range(np.shape(tmp)[0]):
                cv2.line(new_im, (tmp[i, 0], tmp[i, 1]), (tmp[i-1, 0], tmp[i-1, 1]), (204, 0, 255), 10)
        return new_im

    def save_paths(self):
        if Settings.save_paths:
            np.savez('pySonarLog/paths_{}'.format(strftime("%Y%m%d-%H%M%S")), paths=np.array(self.paths), pos=np.array(self.pos))

    def draw_wps_on_grid(self, im, pos):
        wp_list = self.data_storage.get_wps()[0]

        if len(wp_list) > 0:
            vehicle_width = np.round(CollisionSettings.vehicle_margin * 801 / self.range).astype(int)
            if self.path_ok:
                color = PlotSettings.wp_on_grid_color
            else:
                color = (255, 0, 0)
            try:
                wp1_grid = NED2grid(wp_list[0][0], wp_list[0][1], pos[0], pos[1], pos[2], self.range)
                cv2.circle(im, wp1_grid,
                           vehicle_width, color,
                           PlotSettings.wp_on_grid_thickness)
            except ValueError:
                return im
            for i in range(1, len(wp_list)):
                wp_NED, constrained = constrainNED2range((wp_list[i][0], wp_list[i][1]),
                                                         (wp_list[i-1][0], wp_list[i-1][1]),
                                                         pos[0], pos[1], pos[2], self.range)
                wp2_grid = NED2grid(wp_NED[0], wp_NED[1], pos[0], pos[1], pos[2], self.range)
                cv2.circle(im, wp2_grid, vehicle_width, color,
                           PlotSettings.wp_on_grid_thickness)
                cv2.line(im, wp1_grid, wp2_grid, color, vehicle_width)
                if constrained:
                    break
                else:
                    wp1_grid = wp2_grid
        return im


class CollisionStatus(Enum):
    NO_DANGER = 0
    NO_FEASIBLE_ROUTE = 1
    SMOOTH_PATH_VIOLATES_MARGIN = 2
    NEW_ROUTE_OK = 3


class CollisionData:
    pos_lock = threading.Lock()
    obs_lock = threading.Lock()
    wp_lock = threading.Lock()
    lat = 0
    long = 0
    psi = 0
    obstacles = []
    range = 30
    wp_list = []
    wp_counter = 0

    def update_pos(self, lat=None, long=None, psi=None):
        self.pos_lock.acquire(blocking=True)
        if lat is not None:
            self.lat = lat
        if long is not None:
            self.long = long
        if psi is not None:
            self.psi = psi
        self.pos_lock.release()

    def get_pos(self):
        self.pos_lock.acquire()
        tmp = self.lat, self.long, self.psi
        self.pos_lock.release()
        return tmp

    def update_obstacles(self, obstacles, range):
        self.obs_lock.acquire()
        self.obstacles = obstacles
        self.range = range
        self.obs_lock.release()

    def get_obstacles(self):
        self.obs_lock.acquire()
        tmp = self.obstacles, self.range
        self.obs_lock.release()
        return tmp

    def update_wps(self, wp_list, wp_counter):
        self.wp_lock.acquire()
        if wp_list is not None:
            self.wp_list = wp_list
        if wp_counter is not None:
            self.waypoint_counter = wp_counter
        self.wp_lock.release()

    def get_wps(self):
        self.wp_lock.acquire()
        tmp = self.wp_list, self.wp_counter
        self.wp_lock.release()
        return tmp

