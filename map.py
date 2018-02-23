from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from settings import MapSettings
from math import pi

class MapWidget(QWidget):
    pos_ellipse = None
    waypoint_objects = []
    obstacle_list = []

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.scene = QGraphicsScene()
        self.view = QGraphicsView()
        self.init_map()
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.view)
        self.setLayout(main_layout)

        
    def init_map(self):
        if MapSettings.display_grid:
            xmin = -310
            xmax = 420
            ymin = -200
            ymax = 400
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

        self.view.setScene(self.scene)

    def update_pos(self, lat, long, psi):
        # Draw pos
        self.scene.removeItem(self.pos_ellipse)
        self.pos_ellipse = \
            QGraphicsRectItem(long*10 -
                                        (MapSettings.vehicle_size / 2)*MapSettings.vehicle_form_factor,
                                           -lat*10 - MapSettings.vehicle_size / 2,
                                           MapSettings.vehicle_size*MapSettings.vehicle_form_factor,
                                        MapSettings.vehicle_size)
        self.pos_ellipse.setTransformOriginPoint(long*10, -lat*10)
        self.pos_ellipse.setRotation(psi*180.0/pi)
        self.pos_ellipse.setPen(MapSettings.vehicle_pen)
        self.pos_ellipse.setBrush(MapSettings.vehicle_brush)
        self.scene.addItem(self.pos_ellipse)
        
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
                # TODO: Paint obstacles
            except Exception as e:
                print('Add waypoints: {}'.format(e))
