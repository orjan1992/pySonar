import numpy as np
import traceback
def print_args(**kwargs):
    tmp = ''
    for a in kwargs:
        tmp = '{}{}: {}\t'.format(tmp, a, kwargs[a])
    print(tmp)

def grid2vehicle(x_grid, y_grid, range):
    """
    grid zero is upper left corner
    :param x_grid: Horizontal grid coordinates, positive to the right
    :param y_grid: Vertical grid coordinates, positive downwards
    :param range: grid range
    :return: (x_veh, y_veh)
    x_veh is longitudinal vehicle direction, y_veh is transverse
    """
    x_veh = (801 - y_grid) * range / 801.0
    y_veh = (x_grid - 801) * range / 801.0
    return x_veh, y_veh

def grid2vehicle_rad(x_grid, y_grid, range):
    tmp = grid2vehicle(x_grid, y_grid, range)
    return (tmp[0]**2 + tmp[1]**2)**0.5

def vehicle2grid(x_veh, y_veh, range):
    """
    :param x_veh: longitudinal vehicle direction
    :param y_veh: transverse vehicle direction
    :param range: grid range
    :return: (x_grid, y_grid)
    """
    y_grid = int(801 - (x_veh * 801.0 / range))
    # if y_grid < 0:
    #     raise ValueError('y is outside grid')
    x_grid = int(y_veh * 801.0 / range + 801)
    # if x_grid < 0:
    #     raise ValueError('x is outside grid')
    return x_grid, y_grid

def vehicle2NED(x_veh, y_veh, N_veh, E_veh, yaw):
    """
    :param x_veh: x_coord in vehicle frame
    :param y_veh: y_coord in vehicle frame
    :param N_veh: North pos of vehicle
    :param E_veh: East pos of vehicle
    :param yaw: heading of vehicle
    :return: (N, E) translated from x_veh, y_veh
    """
    r = np.sqrt(x_veh**2 + y_veh**2)
    alpha = np.arctan2(y_veh, x_veh)
    north = N_veh + r * np.cos(alpha + yaw)
    east = E_veh + r * np.sin(alpha + yaw)
    return north, east

def NED2vehicle(north, east, N_veh, E_veh, yaw):
    """
    :param north: north coord of pos
    :param east: east coord of pos
    :param N_veh: North pos of vehicle
    :param E_veh: East pos of vehicle
    :param yaw: heading of vehicle
    :return: (x_veh, y_veh) translated from (north, east)
    """

    x = north - N_veh
    y = east - E_veh
    r = np.sqrt(x**2 + y**2)
    alpha = np.arctan2(y, x) - yaw
    x_veh = r*np.cos(alpha)
    y_veh = r*np.sin(alpha)
    return x_veh, y_veh

def grid2NED(x_grid, y_grid, range, N_veh, E_veh, yaw):
    """
    grid zero is upper left corner
    :param x_grid: Horizontal grid coordinates, positive to the right
    :param y_grid: Vertical grid coordinates, positive downwards
    :param range: grid range
    :param N_veh: North pos of vehicle
    :param E_veh: East pos of vehicle
    :param yaw: heading of vehicle
    :return: (N, E_veh) translated from x_veh, y_veh
    """
    x_veh, y_veh = grid2vehicle(x_grid, y_grid, range)
    return vehicle2NED(x_veh, y_veh, N_veh, E_veh, yaw)

def NED2grid(north, east, N_veh, E_veh, yaw, range):
    """
    :param north: north coord of pos
    :param east: east coord of pos
    :param N_veh: North pos of vehicle
    :param E_veh: East pos of vehicle
    :param yaw: heading of vehicle
    :param range: grid range
    :return: (x_grid, y_grid)
    """
    x_veh, y_veh = NED2vehicle(north, east, N_veh, E_veh, yaw)
    return vehicle2grid(x_veh, y_veh, range)

def constrainNED2range(WP, old_WP, N_veh, E_veh, yaw, range):
    """
    :param WP: (N, E)
    :param old_WP: (N_last, E_last)
    :param N_veh: North pos of vehicle
    :param E_veh: East pos of vehicle
    :param yaw: heading of vehicle
    :param range: grid range
    :return: constrained WP in NED frame
    """
    x_veh, y_veh = NED2vehicle(WP[0], WP[1], N_veh, E_veh, yaw)
    if x_veh > range or x_veh < -range or abs(y_veh) > range:
        x_veh_old, y_veh_old = NED2vehicle(old_WP[0], old_WP[1], N_veh, E_veh, yaw)
        alpha = np.arctan2(y_veh - y_veh_old, x_veh - x_veh_old)
        # dist = ((x_veh - x_veh_old)**2 + (y_veh - y_veh_old)**2)**0.5
        # if x_veh > 0:
        if abs(x_veh) > abs(y_veh):
            x_veh_new = range * np.sign(x_veh)
            y_veh_new = y_veh_old + (range - abs(x_veh_old)) * np.sin(alpha)
        else:
            y_veh_new = range * np.sign(y_veh)
            x_veh_new = x_veh_old + (range - abs(y_veh_old)) * np.cos(alpha)
        return vehicle2NED(x_veh_new, y_veh_new, N_veh, E_veh, yaw), True
        # else:
        #     x_veh_new = 0
        #     y_veh_new = y_veh_old + x_veh_old * np.sin(alpha)
        #     return vehicle2NED(x_veh_new, y_veh_new, N_veh, E_veh, yaw), True
    else:
        return WP, False

def NED2grid_with_constraints(WP, pos, range):
    x_veh, y_veh = NED2vehicle(WP[0], WP[1], pos[0], pos[1], pos[2])
    g = vehicle2grid(x_veh, y_veh, range)
    return g, not (abs(x_veh) <= range) and (abs(y_veh) <= range)

def ned2constrained_grid(wp1, wp0, pos, range):
    # 0= not constrained, 1 constrained normal way, 2 constrained inverse
    status = 0
    wp_NED, constrained = constrainNED2range((wp1[0], wp1[1]),
                                             (wp0[0], wp0[1]),
                                             pos[0], pos[1], pos[2], range)
    if constrained:
        status = 1
        wp_NED_extra, constrained = constrainNED2range((wp0[0], wp0[1]),
                                                 (wp1[0], wp1[1]),
                                                 pos[0], pos[1], pos[2], range)
        if constrained:
            status = 2
            wp_grid_extra = NED2grid(wp_NED_extra[0], wp_NED_extra[1], pos[0], pos[1], pos[2], range)
            return wp_grid_extra, status
    wp_grid = NED2grid(wp_NED[0], wp_NED[1], pos[0], pos[1], pos[2], range)
    return wp_grid, status

def norm_cross_product(wp_list):
    c1 = np.array([wp_list[1][0]-wp_list[0][0], wp_list[1][1]-wp_list[0][0]], dtype=float)
    n1 = np.linalg.norm(c1)
    if n1 == 0:
        raise ValueError('Zero norm')
    c1 /= n1
    c2 = np.array([wp_list[2][0]-wp_list[0][0], wp_list[2][1]-wp_list[0][0]], dtype=float)
    n2 = np.linalg.norm(c2)
    if n2 == 0:
        raise ValueError('Zero norm')
    c2 /= n2
    return np.cross(c1, c2)

def angle_diff(wp_list):
    c1 = np.array([wp_list[1][0] - wp_list[0][0], wp_list[1][1] - wp_list[0][0]], dtype=float)
    c2 = np.array([wp_list[2][0] - wp_list[0][0], wp_list[2][1] - wp_list[0][0]], dtype=float)
    alpha1 = np.arctan2(c1[1], c1[0])
    alpha2 = np.arctan2(c2[1], c2[0])
    return abs(wrapToPi(alpha2 - alpha1))

def sat2uint(val, sat):
    if val < 0:
        return 0
    if val > sat:
        return int(sat)
    else:
        return int(val)

# def wrapTo2Pi(angle):
#     positiveInput = (angle > 0)
#     angle = angle % 2 * np.pi
#     if angle == 0 and positiveInput:
#         return 2*np.pi
#     else:
#         return angle
#
# def wrapToPi(angle):
#     if angle < -np.pi or np.pi < angle:
#         return wrapTo2Pi(angle + np.pi) - np.pi
#     else:
#         return angle
#
# def wrapToPiHalf(angle):
#     angle = np.abs(wrapToPi(angle))
#     if angle > np.pi/2:
#         return np.pi - angle
#     else:
#         return angle

def wrapTo2Pi(angle):
    if angle is None:
        return
    positiveInput = (angle > 0)
    angle = np.remainder(angle, 2 * np.pi)
    mask = np.logical_and(angle == 0, positiveInput)
    if np.any(mask):
        try:
            angle[mask] = 2*np.pi
        except TypeError:
            angle = 2*np.pi
    return angle

def wrapToPi(angle):
    if angle is None:
        return
    mask = np.logical_or(angle < -np.pi, angle > np.pi)
    if np.any(mask):
        try:
            angle[mask] = wrapTo2Pi(angle[mask] + np.pi) - np.pi
        except TypeError:
            angle = wrapTo2Pi(angle + np.pi) - np.pi
    return angle

def wrapToPiHalf(angle):
    if angle is None:
        return
    angle = np.abs(wrapToPi(angle))
    mask = angle > np.pi / 2
    if np.any(mask):
        try:
            angle[mask] = np.pi - angle[mask]
        except TypeError:
            angle = np.pi - angle
    return angle

def path_length(wp_list):
    length = 0
    for wp1, wp2 in zip(wp_list[:-1], wp_list[1:]):
        length += ((wp2[0]-wp1[0])**2 + (wp2[1]-wp1[1])**2)**0.5
    return length

def in_interval(intervals, integer):
    for i,j in zip(intervals[:-1], intervals[1:]):
        if i <= integer < j:
            return intervals.index(i)
    return -1

if __name__ == '__main__':
    N_veh = 0
    E_veh = 0
    yaw = 0
    print(constrainNED2range((60, 60), (0, 0), 0, 0, 0, 30))
