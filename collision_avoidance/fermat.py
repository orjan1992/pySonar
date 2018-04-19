import numpy as np
from settings import CollisionSettings
import logging
logger = logging.getLogger('Fermat')

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
                new_list.append([x, y, wp2[2], wp2[3]])
            tmp_list = []
            for theta in np.linspace(0, theta_end, step + 1, endpoint=True):
                x = p_end[0] - kappa * (theta ** 0.5) * np.cos(-rho * theta + chi_end)
                y = p_end[1] - kappa * (theta ** 0.5) * np.sin(-rho * theta + chi_end)
                tmp_list.append([x, y, wp3[2], wp3[3]])
        except ValueError as e:
            logger.error('step={}\tdelta_chi_mag={}'.format(delta_chi_mag / CollisionSettings.fermat_step_factor, delta_chi_mag))
        tmp_list.reverse()
        new_list.extend(tmp_list)
    new_list.append(wp_list[-1])
    return new_list


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # wp_list = [(0, 0, 0, 0), (10, 6, 0, 0), (4, 8, 0, 0), (15, 16, 0, 0), (6, 18, 0, 0), (17, 21, 0, 0)]
    wp_list = [(3, 10, 0, 0), (5, 5, 0, 0), (7.5, 15, 0, 0), (15, 15, 0, 0), (15, 6, 0, 0)]
    # wp_list.reverse()
    wp_array = np.array(wp_list)

    smooth_wp = fermat(wp_list)
    smooth_wp_array = np.array(smooth_wp)
    plt.plot(wp_array[:, 0], wp_array[:, 1], 'b')
    plt.plot(smooth_wp_array[:, 0], smooth_wp_array[:, 1], 'r')
    # plt.xlim([6, 12])
    # plt.ylim([12, 16])
    plt.show()