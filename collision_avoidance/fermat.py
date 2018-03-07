import numpy as np
from settings import CollisionSettings


def fermat(wp_list):
    new_list = [wp_list[0][:2]]
    wp_array = np.array(wp_list)[:, :2]
    for wp1, wp2, wp3 in zip(wp_array, wp_array[1:], wp_array[2:]):
        # Segment lengths
        l_in = np.linalg.norm(wp2 - wp1)
        l_out = np.linalg.norm(wp3 - wp2)
        # Endpoint tangents
        v_in = (wp2 - wp1) / l_in
        v_out = (wp3 - wp2) / l_out
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
                    2 * (-2 / (4 * theta ** 2 + 1) - 1) ** 2 - (chi_theta_max - theta - np.arctan(2 * theta)) * (
                    16 * theta / (4 * theta ** 2 + 1) ** 2))
        theta_end = theta
        # find maximum curvature
        theta_kappa_max = np.min([theta_end, ((7 ** 0.5) / 2 - 5 / 4) ** 0.5])
        print(np.argmin([theta_end, ((7 ** 0.5) / 2 - 5 / 4) ** 0.5]))

        # compute scaling factor
        kappa = (1 / CollisionSettings.kappa_max) * (2 * (theta_kappa_max ** 0.5) * (3 + 4 * (theta_kappa_max ** 2))) / \
                (1 + 4 * (theta_kappa_max ** 2)) ** 1.5

        # find shortest segment and intersection point with both segments
        # if l_in < l_out:
        #     p_0 = wp1 + 0.5 * l_in * v_in
        #     p_end = wp2 + ((-p_0 + wp2) / v_in) * v_out
        # else:
        #     if v_out[1] == 0:
        #         v_out[1] = 10**(-3)
        #     p_end = wp3 - 0.5 * l_out * v_out
        #     p_0 = wp2 - ((-p_end - wp2) / v_out) * v_in


        l_1 = kappa * (theta_end ** 0.5) * np.cos(theta_end)
        h = kappa * (theta_end ** 0.5) * np.sin(theta_end)
        alpha = (np.pi - delta_chi_mag) / 2
        l_2 = h / np.tan(alpha)
        l = l_1 + l_2
        p_0 = wp2 - v_in*l
        p_end = wp2 + v_out*l
        # new_list.append(p_0)
        # new_list.append(p_end)


        # # Find start and end course angles
        chi_0 = np.arctan2(v_in[1], v_in[0])
        chi_end = np.arctan2(v_out[1], v_out[0])

        for theta in np.linspace(0, theta_end, 5, endpoint=False):
            x = p_0[0] + kappa * (theta ** 0.5) * np.cos(rho * theta + chi_0)
            y = p_0[1] + kappa * (theta ** 0.5) * np.sin(rho * theta + chi_0)
            new_list.append([x, y])
        # i = len(new_list) - 1
        tmp_list = []
        for theta in np.linspace(0, theta_end, 5, endpoint=True):
            x = p_end[0] - kappa * (theta ** 0.5) * np.cos(-rho * theta + chi_end)
            y = p_end[1] - kappa * (theta ** 0.5) * np.sin(-rho * theta + chi_end)
            tmp_list.append([x, y])
        tmp_list.reverse()
        new_list.extend(tmp_list)
    new_list.append(wp_list[-1])
    return new_list


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    wp_list = [[0, 0], [10, 6], [4, 8], [15, 16], [6, 18], [17, 21]]
    # wp_list = [[0, 0], [5, 5], [7.5, 15], [15, 15]]
    wp_array = np.array(wp_list)

    smooth_wp = fermat(wp_list)
    smooth_wp_array = np.array(smooth_wp)
    plt.plot(wp_array[:, 0], wp_array[:, 1], 'r')
    plt.plot(smooth_wp_array[:, 0], smooth_wp_array[:, 1], 'b')
    # plt.xlim([6, 12])
    # plt.ylim([12, 16])
    plt.show()
    print(smooth_wp)