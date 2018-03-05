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

def vehicle2grid(x_veh, y_veh, range):
    """
    :param x_veh: longitudinal vehicle direction
    :param y_veh: transverse vehicle direction
    :param range: grid range
    :return: (x_grid, y_grid)
    """
    y_grid = int(801 - (x_veh * 801.0 / range))
    if y_grid < 0:
        print('y_grid truncated to zero, old_value: {}'.format(y_grid))
        traceback.print_stack()
        y_grid = 0
    x_grid = int(y_veh * 801.0 / range + 801)
    if x_grid < 0:
        print('x_grid truncated to zero, old_value: {}'.format(x_grid))
        traceback.print_stack()
        x_grid = 0
    return x_grid, y_grid

def vehicle2NED(x_veh, y_veh, N_veh, E_veh, psi):
    """
    :param x_veh: x_coord in vehicle frame
    :param y_veh: y_coord in vehicle frame
    :param N_veh: North pos of vehicle
    :param E_veh: East pos of vehicle
    :param psi: heading of vehicle
    :return: (N, E) translated from x_veh, y_veh
    """
    r = np.sqrt(x_veh**2 + y_veh**2)
    alpha = np.arctan2(y_veh, x_veh)
    north = N_veh + r * np.cos(alpha + psi)
    east = E_veh + r * np.sin(alpha + psi)
    return north, east

def NED2vehicle(north, east, N_veh, E_veh, psi):
    """
    :param north: north coord of pos
    :param east: east coord of pos
    :param N_veh: North pos of vehicle
    :param E_veh: East pos of vehicle
    :param psi: heading of vehicle
    :return: (x_veh, y_veh) translated from (north, east)
    """

    x = north - N_veh
    y = east - E_veh
    r = np.sqrt(x**2 + y**2)
    alpha = np.arctan2(y, x) - psi
    x_veh = r*np.cos(alpha)
    y_veh = r*np.sin(alpha)
    return x_veh, y_veh

def grid2NED(x_grid, y_grid, range, N_veh, E_veh, psi):
    """
    grid zero is upper left corner
    :param x_grid: Horizontal grid coordinates, positive to the right
    :param y_grid: Vertical grid coordinates, positive downwards
    :param range: grid range
    :param N_veh: North pos of vehicle
    :param E_veh: East pos of vehicle
    :param psi: heading of vehicle
    :return: (N, E_veh) translated from x_veh, y_veh
    """
    x_veh, y_veh = grid2vehicle(x_grid, y_grid, range)
    return vehicle2NED(x_veh, y_veh, N_veh, E_veh, psi)

def NED2grid(north, east, N_veh, E_veh, psi, range):
    """
    :param north: north coord of pos
    :param east: east coord of pos
    :param N_veh: North pos of vehicle
    :param E_veh: East pos of vehicle
    :param psi: heading of vehicle
    :param range: grid range
    :return: (x_grid, y_grid)
    """
    x_veh, y_veh = NED2vehicle(north, east, N_veh, E_veh, psi)
    return vehicle2grid(x_veh, y_veh, range)

def constrainNED2range(WP, N_veh, E_veh, psi, range):
    """
    :param WP: (N, E)
    :param old_WP: (N_last, E_last)
    :param N_veh: North pos of vehicle
    :param E_veh: East pos of vehicle
    :param psi: heading of vehicle
    :param range: grid range
    :return: constrained WP in NED frame
    """
    x_veh, y_veh = NED2vehicle(WP[0], WP[1], N_veh, E_veh, psi)
    if abs(x_veh) > range or abs(y_veh) > range:
        alpha = np.arctan2(y_veh, x_veh)
        if abs(x_veh) > abs(y_veh):
            x_veh_new = range * np.sign(x_veh)
            y_veh_new = x_veh_new * np.tan(alpha)
        else:
            y_veh_new = range * np.sign(y_veh)
            x_veh_new = y_veh_new / np.tan(alpha)
        return vehicle2NED(x_veh_new, y_veh_new, N_veh, E_veh, psi), True
    else:
        return WP, False

def sat2uint(val, sat):
    if val < 0:
        return 0
    if val > sat:
        return int(sat)
    else:
        return int(val)


if __name__ == '__main__':
    N_veh = 3.51
    E_veh = 4.3
    psi = np.pi/6
    x_veh = 1.67
    y_veh = 1.57
    range = 1.57
    # N, E = vehicle2NED(x_veh, y_veh, N_veh, E_veh, psi)
    # print_args(N=N, E=E)
    # x_veh, y_veh = NED2vehicle(N, E, N_veh, E_veh, psi)
    # print_args(x_veh=x_veh, y_veh=y_veh)
    N, E = grid2NED(801, 801, 30, N_veh, E_veh, psi)
    print_args(N=N, E=E)