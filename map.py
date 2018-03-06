from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from settings import MapSettings
import math
from coordinate_transformations import *

class MapWidget(QWidget):
    pos_ellipse = None
    waypoint_objects = []
    avoidance_waypoints = []
    obstacle_list = []

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.scene = QGraphicsScene()
        self.view = MyQGraphicsView()
        self.init_map()
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.view)
        self.setLayout(main_layout)

    def init_map(self):
        if MapSettings.display_grid:
            xmin = -3100
            xmax = 4200
            ymin = -2000
            ymax = 4000
            for i in range(xmin, xmax, MapSettings.grid_dist):
                self.scene.addLine(i, ymin, i, ymax, MapSettings.grid_pen)
            for i in range(ymin, ymax, MapSettings.grid_dist):
                self.scene.addLine(xmin, i, xmax, i, MapSettings.grid_pen)
            self.scene.addLine(-MapSettings.grid_dist, 0, MapSettings.grid_dist,
                               0, MapSettings.grid_center_pen)
            self.scene.addLine(0, -MapSettings.grid_dist, 0, MapSettings.grid_dist,
                               MapSettings.grid_center_pen)

        self.scene.addEllipse(108.0412 - 5, -133.7201 - 5, 10, 10,
                              MapSettings.obstacle_pen, MapSettings.obstacle_brush)
        self.scene.addEllipse(-53.5571 - 24.88 / 2, -60.9444 - 5, 24.88, 10,
                              MapSettings.obstacle_pen, MapSettings.obstacle_brush)
        self.scene.addEllipse(-214.7458 - 5, 37.2886 - 40, 10, 80,
                              MapSettings.obstacle_pen, MapSettings.obstacle_brush)

        self.scene.addRect(101.6381 - 7.5, 31.0354 - 13.3 / 2, 15, 13.3,
                           MapSettings.obstacle_pen, MapSettings.obstacle_brush)

        self.scene.addRect(-2.0295 - 5, 120.6624 - 5, 10, 10,
                           MapSettings.obstacle_pen, MapSettings.obstacle_brush)
        self.scene.addRect(311.4198 - 5, 120.6624 - 50, 10, 100,
                           MapSettings.obstacle_pen, MapSettings.obstacle_brush)
        self.scene.addRect(59.9079 - 100, 406.9405 - 5, 200, 10,
                           MapSettings.obstacle_pen, MapSettings.obstacle_brush)
        self.scene.addRect(-2.0295 - 50, -211.9193 - 5, 100, 10,
                           MapSettings.obstacle_pen, MapSettings.obstacle_brush)

        self.pos_ellipse = QGraphicsEllipseItem(0, 0, MapSettings.vehicle_size,
                                                MapSettings.vehicle_size)
        self.pos_ellipse.setPen(MapSettings.vehicle_pen)
        self.scene.addItem(self.pos_ellipse)

        self.sonar_circle = QGraphicsEllipseItem(0, 0, MapSettings.vehicle_size,
                                                MapSettings.vehicle_size)
        self.sonar_circle.setPen(MapSettings.vehicle_pen)
        self.scene.addItem(self.sonar_circle)

        self.view.setScene(self.scene)
        self.view.scale(1.4, 1.4)
        self.view.centerOn(0, 0)

    def update_pos(self, lat, long, psi, range):
        # Draw pos
        self.scene.removeItem(self.pos_ellipse)
        self.pos_ellipse = \
            QGraphicsRectItem(long*10 -
                                        (MapSettings.vehicle_size / 2)*MapSettings.vehicle_form_factor,
                                           -lat*10 - MapSettings.vehicle_size / 2,
                                           MapSettings.vehicle_size*MapSettings.vehicle_form_factor,
                                        MapSettings.vehicle_size)
        self.pos_ellipse.setTransformOriginPoint(long*10, -lat*10)
        self.pos_ellipse.setRotation(psi*180.0/math.pi)
        self.pos_ellipse.setPen(MapSettings.vehicle_pen)
        self.pos_ellipse.setBrush(MapSettings.vehicle_brush)
        self.scene.addItem(self.pos_ellipse)

        # draw sonar circle
        self.scene.removeItem(self.sonar_circle)
        self.sonar_circle = QGraphicsEllipseItem(long*10 - range*10, -lat*10 - range*10, 2*range*10, 2*range*10)
        self.sonar_circle.setTransformOriginPoint(long*10, -lat*10)
        self.sonar_circle.setRotation(psi*180.0/math.pi)
        self.sonar_circle.setSpanAngle(180*16)
        self.sonar_circle.setBrush(MapSettings.sonar_circle_brush)
        self.scene.addItem(self.sonar_circle)

    def update_avoidance_waypoints(self, waypoints):
        try:
            # remove old waypoints
            for obj in self.avoidance_waypoints:
                self.scene.removeItem(obj)
            self.avoidance_waypoints.clear()
        except Exception as e:
            print('Remove waypoints: {}'.format(e))

        # draw new
        if len(waypoints) > 0:
            p = QGraphicsEllipseItem(waypoints[0][1] * 10 - MapSettings.waypoint_size / 2,
                                               -waypoints[0][0] * 10 - MapSettings.waypoint_size / 2,
                                               MapSettings.waypoint_size,
                                               MapSettings.waypoint_size)

            p.setPen(MapSettings.avoidance_waypoint_pen)
            self.scene.addItem(p)
            self.avoidance_waypoints.append(p)
            try:
                for i in range(1, len(waypoints)):
                    p = QGraphicsEllipseItem(waypoints[i][1] * 10 - MapSettings.waypoint_size / 2,
                                                       -waypoints[i][0] * 10 - MapSettings.waypoint_size / 2,
                                                       MapSettings.waypoint_size,
                                                       MapSettings.waypoint_size)

                    p.setPen(MapSettings.avoidance_waypoint_pen)
                    self.scene.addItem(p)
                    self.avoidance_waypoints.append(p)

                    l = QGraphicsLineItem(waypoints[i][1] * 10,
                                                    -waypoints[i][0] * 10,
                                                    waypoints[i - 1][1] * 10,
                                                    -waypoints[i - 1][0] * 10)
                    l.setPen(MapSettings.avoidance_waypoint_pen)
                    self.scene.addItem(l)
                    self.avoidance_waypoints.append(l)
            except Exception as e:
                print('Add waypoints: {}'.format(e))

    def update_waypoints(self, waypoints, waypoint_counter):
        try:
            # remove old waypoints
            for obj in self.waypoint_objects:
                self.scene.removeItem(obj)
            self.waypoint_objects.clear()
        except Exception as e:
            print('Remove waypoints: {}'.format(e))

        # draw new
        if len(waypoints) > 0:
            p = QGraphicsEllipseItem(waypoints[0][1] * 10 - MapSettings.waypoint_size / 2,
                                     -waypoints[0][0] * 10 - MapSettings.waypoint_size / 2,
                                     MapSettings.waypoint_size,
                                     MapSettings.waypoint_size)
            if waypoint_counter == 1:
                p.setPen(MapSettings.waypoint_active_pen)
            else:
                p.setPen(MapSettings.waypoint_inactive_pen)
            self.scene.addItem(p)
            self.waypoint_objects.append(p)
            try:
                for i in range(1, len(waypoints)):
                    p = QGraphicsEllipseItem(waypoints[i][1] * 10 - MapSettings.waypoint_size / 2,
                                             -waypoints[i][0] * 10 - MapSettings.waypoint_size / 2,
                                             MapSettings.waypoint_size,
                                             MapSettings.waypoint_size)
                    if waypoint_counter == i or waypoint_counter == i + 1:
                        p.setPen(MapSettings.waypoint_active_pen)
                    else:
                        p.setPen(MapSettings.waypoint_inactive_pen)
                    self.scene.addItem(p)
                    self.waypoint_objects.append(p)

                    l = QGraphicsLineItem(waypoints[i][1] * 10,
                                          -waypoints[i][0] * 10,
                                          waypoints[i - 1][1] * 10,
                                          -waypoints[i - 1][0] * 10)
                    if waypoint_counter == i:
                        l.setPen(MapSettings.waypoint_active_pen)
                    else:
                        l.setPen(MapSettings.waypoint_inactive_pen)
                    self.scene.addItem(l)
                    self.waypoint_objects.append(l)
            except Exception as e:
                print('Add waypoints: {}'.format(e))

    def update_obstacles(self, obstacles, range, lat, long, psi):
        for ellipse in self.obstacle_list:
            self.scene.removeItem(ellipse)
        self.obstacle_list.clear()

        for obs in obstacles:
            # x, y, r1, r2 = self.sonar_coord2scene(obs, range, lat, long, psi)
            # ellipse = QGraphicsEllipseItem(x, y, r1, r2)
            # ellipse.setPen(MapSettings.sonar_obstacle_pen)
            # ellipse.setTransformOriginPoint(x + r1/2, y + r2/2)
            # ellipse.setRotation(psi*180.0/math.pi + obs[2])
            # self.scene.addItem(ellipse)
            # self.obstacle_list.append(ellipse)
            polygon = QPolygonF()
            for p in obs:
                N, E = grid2NED(p[0][0], p[0][1], range, lat, long, psi)
                polygon.append(QPointF(E*10.0, -N*10.0))
            q_poly = QGraphicsPolygonItem()
            q_poly.setPolygon(polygon)
            q_poly.setPen(MapSettings.sonar_obstacle_pen)
            self.scene.addItem(q_poly)
            self.obstacle_list.append(q_poly)


    @staticmethod
    def sonar_coord2scene(obs, range_scale, lat, long, psi):
        x_obs = (obs[0][0] - 801) * range_scale / 801
        y_obs = (801 - obs[0][1]) * range_scale / 801
        alpha = math.atan2(x_obs, y_obs) + psi
        r1 = obs[1][0] * range_scale / 801
        r2 = obs[1][1] * range_scale / 801
        r = math.sqrt(x_obs**2 + y_obs**2)
        x_scene = long + r*math.sin(alpha) - r1/2
        y_scene = -lat - r*math.cos(alpha) - r2/2
        return x_scene*10, y_scene*10, r1*10, r2*10



class MyQGraphicsView(QGraphicsView):

    def __init__(self, parent=None):
        super(MyQGraphicsView, self).__init__ (parent)
        self.setDragMode(1)

    def wheelEvent(self, event):
        """
        Zoom in or out of the view.
        """
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor

        # Save the scene pos
        old_pos = self.mapToScene(event.pos())

        # Zoom
        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor
        self.scale(zoom_factor, zoom_factor)

        # Get the new position
        new_pos = self.mapToScene(event.pos())

        # Move scene to old position
        delta = new_pos - old_pos
        self.translate(delta.x(), delta.y())
