import numpy as np
from settings import CollisionSettings

THETA_MAX = ((7 ** 0.5) / 2 - 5 / 4) ** 0.5

def fermat(wp_list):
    new_list = [wp_list[0][:2]]
    wp_array = np.array(wp_list)
    for wp1, wp2, wp3 in zip(wp_array, wp_array[1:], wp_array[2:]):
        # Segment lengths
        l_in = np.linalg.norm(wp2[:2] - wp1[:2])
        l_out = np.linalg.norm(wp3[:2] - wp2[:2])
        # Endpoint tangents
        v_in = (wp2[:2] - wp1[:2]) / l_in
        v_out = (wp3[:2] - wp2[:2]) / l_out
        # Course change magnitude
        delta_chi_mag = np.arccos(np.dot(v_in, v_out))
        # Course change direction
        rho = -np.sign(v_in[1] * v_out[0] - v_in[0] * v_out[1])
        delta_chi = delta_chi_mag * rho
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
        theta_kappa_max = np.min([theta_end, THETA_MAX])

        # compute scaling factor
        kappa = (1 / CollisionSettings.fermat_kappa_max) * (2 * (theta_kappa_max ** 0.5) * (3 + 4 * (theta_kappa_max ** 2))) / \
                (1 + 4 * (theta_kappa_max ** 2)) ** 1.5

        # find shortest segment and intersection point with both segments
        l = kappa * (theta_end ** 0.5) * (np.cos(theta_end) + np.sin(theta_end) / np.tan((np.pi - delta_chi_mag) / 2))
        p_0 = wp2[:2] - v_in*l
        p_end = wp2[:2] + v_out*l


        # # Find start and end course angles
        chi_0 = np.arctan2(v_in[1], v_in[0])
        chi_end = np.arctan2(v_out[1], v_out[0])

        step = np.round(delta_chi_mag / CollisionSettings.fermat_step_factor).astype(int)

        # Find intermediate waypoints
        for theta in np.linspace(0, theta_end, step, endpoint=False):
            x = p_0[0] + kappa * (theta ** 0.5) * np.cos(rho * theta + chi_0)
            y = p_0[1] + kappa * (theta ** 0.5) * np.sin(rho * theta + chi_0)
            new_list.append([x, y, wp2[2], wp2[3]])
        tmp_list = []
        for theta in np.linspace(0, theta_end, step + 1, endpoint=True):
            x = p_end[0] - kappa * (theta ** 0.5) * np.cos(-rho * theta + chi_end)
            y = p_end[1] - kappa * (theta ** 0.5) * np.sin(-rho * theta + chi_end)
            tmp_list.append([x, y, wp3[2], wp3[3]])
        tmp_list.reverse()
        new_list.extend(tmp_list)
    new_list.append(wp_list[-1])
    return new_list


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # wp_list = [[0, 0], [10, 6], [4, 8], [15, 16], [6, 18], [17, 21]]
    wp_list = [[0, 0], [5, 5], [7.5, 15], [15, 15]]
    wp_array = np.array(wp_list)

    smooth_wp = fermat(wp_list)
    smooth_wp_array = np.array(smooth_wp)
    double_smooth = fermat(smooth_wp)
    double_smooth_array = np.array(double_smooth)
    plt.plot(wp_array[:, 0], wp_array[:, 1], 'r')
    plt.plot(smooth_wp_array[:, 0], smooth_wp_array[:, 1], 'b')
    plt.plot(double_smooth_array[:, 0], double_smooth_array[:, 1], 'g')
    # plt.xlim([6, 12])
    # plt.ylim([12, 16])
    plt.show()