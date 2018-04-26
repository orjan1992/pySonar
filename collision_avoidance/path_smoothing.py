import numpy as np
from settings import CollisionSettings
import logging
from scipy.interpolate import CubicSpline
logger = logging.getLogger('PathSmoothing')

THETA_MAX = ((7 ** 0.5) / 2 - 5 / 4) ** 0.5

def fermat(wp_list):
    new_list = [wp_list[0]]
    wp_array = np.array(wp_list)
    for wp1, wp2, wp3 in zip(wp_array, wp_array[1:], wp_array[2:]):
        # Segment lengths
        l_in = np.linalg.norm(wp2[:2] - wp1[:2])
        l_out = np.linalg.norm(wp3[:2] - wp2[:2])
        if l_in == 0 or l_out == 0:
            new_list.append(wp2)
            continue
        # Endpoint tangents
        v_in = (wp2[:2] - wp1[:2]) / l_in
        v_out = (wp3[:2] - wp2[:2]) / l_out
        # Course change magnitude
        # TODO: Should do something if this is zero
        delta_chi_mag = np.arccos(np.clip(np.dot(v_in, v_out), -1, 1))
        # Course change direction
        rho = -np.sign(v_in[1] * v_out[0] - v_in[0] * v_out[1])
        # delta_chi = delta_chi_mag * rho
        # Endpoint tanget
        chi_theta_max = delta_chi_mag / 2

        # find end curvature numerically
        theta = np.pi / 6
        while np.abs(chi_theta_max - theta - np.arctan(2 * theta)) > 10 ** (-3):
            theta = theta - (2 * (chi_theta_max - theta - np.arctan(2 * theta)) * (-2 / (4 * theta ** 2 + 1) - 1)) / (
                    2 * (-2 / (4 * theta ** 2 + 1) - 1) ** 2 - (chi_theta_max - theta - np.arctan(2 * theta)) *
                    (16 * theta / (4 * theta ** 2 + 1) ** 2))
        theta_end = theta
        # find maximum curvature
        # theta_kappa_max = np.min([theta_end, THETA_MAX])

        # compute scaling factor
        # kappa = (1 / CollisionSettings.fermat_kappa_max) * (2 * (theta_kappa_max ** 0.5) * (3 + 4 * (theta_kappa_max ** 2))) / \
        #         (1 + 4 * (theta_kappa_max ** 2)) ** 1.5

        # find shortest segment and intersection point with both segments
        l = np.min([l_in / 2, l_out / 2])
        p_0 = wp2[:2] - v_in*l
        p_end = wp2[:2] + v_out*l
        numerator = np.tan((np.pi - delta_chi_mag) / 2)
        if numerator != 0:
            numerator = (theta_end ** 0.5) * (np.cos(theta_end) + np.sin(theta_end) / numerator)
            if numerator != 0:
                kappa = l / numerator
        if kappa is None:
            return wp_list

        # # Find start and end course angles
        chi_0 = np.arctan2(v_in[1], v_in[0])
        chi_end = np.arctan2(v_out[1], v_out[0])

        step = np.round(delta_chi_mag / CollisionSettings.fermat_step_factor).astype(int)

        # Find intermediate waypoints
        try:
            for theta in np.linspace(0, theta_end, step, endpoint=False):
                x = p_0[0] + kappa * (theta ** 0.5) * np.cos(rho * theta + chi_0)
                y = p_0[1] + kappa * (theta ** 0.5) * np.sin(rho * theta + chi_0)
                new_list.append([x, y, wp2[2]])
            tmp_list = []
            for theta in np.linspace(0, theta_end, step + 1, endpoint=True):
                x = p_end[0] - kappa * (theta ** 0.5) * np.cos(-rho * theta + chi_end)
                y = p_end[1] - kappa * (theta ** 0.5) * np.sin(-rho * theta + chi_end)
                tmp_list.append([x, y, wp3[2]])
        except ValueError as e:
            logger.error('step={}\tdelta_chi_mag={}'.format(delta_chi_mag / CollisionSettings.fermat_step_factor, delta_chi_mag))
        tmp_list.reverse()
        new_list.extend(tmp_list)
    new_list.append(wp_list[-1])
    return new_list

def cubic_path(wp_list):
    if len(wp_list) <= 1:
        logger.info('Too few waypoints')
        return wp_list
    wp_array = np.array(wp_list)
    omega = np.arange(0, len(wp_list), 1, int)

    xpath = CubicSpline(omega, wp_array[:, 0])
    ypath = CubicSpline(omega, wp_array[:, 1])
    zpath = CubicSpline(omega, wp_array[:, 2])
    omega_discrete = np.arange(0, len(wp_list)-.999, CollisionSettings.cubic_smoothing_discrete_step)
    wp_array_discrete = np.zeros((len(omega_discrete), 3))
    wp_array_discrete[:, 0] = xpath(omega_discrete)
    wp_array_discrete[:, 1] = ypath(omega_discrete)
    wp_array_discrete[:, 2] = zpath(omega_discrete)
    return np.ndarray.tolist(wp_array_discrete)

def path_grad(wp_list):
    if len(wp_list) > 2:
        wp_array = np.array(wp_list)
        dist = np.sqrt(np.square(wp_array[1:, 0] - wp_array[:-1, 0]) + np.square(wp_array[1:, 1] - wp_array[:-1, 1]))
        tmp = dist == 0
        if np.any(tmp):
            mask = np.logical_not(tmp)
            dist = dist[mask]
            wp_array = wp_array[np.append(mask, True), :]
            pop_ind = np.nonzero(tmp)[0]
            for i in pop_ind:
                wp_list.pop(i)
        dist_cum = np.cumsum(dist)
        theta = np.arctan2(wp_array[1:, 1] - wp_array[:-1, 1], wp_array[1:, 0] - wp_array[:-1, 0])
        grad = np.gradient(np.unwrap(theta, discont=np.pi / 2), dist_cum)
        return wp_list, np.abs(grad)
    else:
        return wp_list, np.array([0])

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # # wp_list = [(0, 0, 0, 0), (10, 6, 0, 0), (4, 8, 0, 0), (15, 16, 0, 0), (6, 18, 0, 0), (17, 21, 0, 0)]
    # wp_list = [(3, 10, 0, 0), (5, 5, 0, 0), (7.5, 15, 0, 0), (15, 15, 0, 0), (15, 6, 0, 0)]
    # # wp_list.reverse()
    # wp_array = np.array(wp_list)
    #
    # smooth_wp = fermat(wp_list)
    # smooth_wp_array = np.array(smooth_wp)
    # plt.plot(wp_array[:, 0], wp_array[:, 1], 'b')
    # plt.plot(smooth_wp_array[:, 0], smooth_wp_array[:, 1], 'r')
    # cubic_array = np.array(cubic_path(wp_list))
    # plt.plot(cubic_array[:, 0], cubic_array[:, 1], 'k')
    #
    # # plt.xlim([6, 12])
    # # plt.ylim([12, 16])
    # plt.show()

    from scipy.io import loadmat
    from scipy.interpolate import interp1d
    wp_list = np.ndarray.tolist(loadmat('pySonarLog/paths_20180426-095740.mat')['paths'][0])
    wp_list, grad = path_grad(wp_list)
    wp_array = np.array(wp_list)
    plt.figure(1)
    plt.plot(wp_array[:, 0], wp_array[:, 1])
    plt.figure(2)
    plt.plot(grad*180.0/np.pi)
    plt.show()


    # dist = np.sqrt(np.square(wp_list[1:, 0] - wp_list[:-1, 0]) + np.square(wp_list[1:, 1] - wp_list[:-1, 1]))
    # tmp = dist == 0
    # if np.any(tmp):
    #     mask = np.logical_not(tmp)
    #     dist = dist[mask]
    #     wp_list = wp_list[np.append(mask, True), :]
    # dist_cum = np.cumsum(dist)
    # theta = np.arctan2(wp_list[1:, 1] - wp_list[:-1, 1], wp_list[1:, 0] - wp_list[:-1, 0])
    # theta_un = np.unwrap(theta, discont=np.pi/2)
    # grad = np.gradient(theta_un, dist_cum)
    # # theta_int = interp1d(dist_cum, theta)
    #
    # plt.figure(1)
    # plt.plot(theta)
    # plt.plot(theta_un)
    # plt.figure(2)
    # plt.plot(wp_list[:, 0], wp_list[:, 1], '-o')
    # plt.figure(3)
    # plt.plot(grad)
    # plt.figure(4)
    # plt.plot(dist_cum)
    # plt.show()
    # # grad = np.gradient()
    # # plt.plot(wp_list[:, 0], wp_list[:, 1])
    # # plt.show()
