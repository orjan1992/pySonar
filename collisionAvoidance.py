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
