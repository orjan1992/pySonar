from settings import LosSettings
import numpy as np
from collision_avoidance.path_smoothing import fermat, path_grad
from pySonarLog.save.path_curvature_plot.referencemodel import ReferenceModel

def turn_vel(wp_grad, i, segment_lengths):
    
    if wp_grad[i] == 0 and len(wp_list) > i + 2:
        surge_speed = LosSettings.cruise_speed
        return LosSettings.cruise_speed, -1
    elif len(wp_list) <= i + 2:
        turn_velocity = 0
    else:
        turn_velocity = LosSettings.safe_turning_speed / (8 * wp_grad[i])


    diff = LosSettings.cruise_speed - turn_velocity
    dist = -1
    if diff > 0:
        t = diff / LosSettings.max_acc
        dist = LosSettings.cruise_speed * t + 0.5 * LosSettings.max_acc * t ** 2
    return turn_velocity, dist

def segment_length(wp1, wp2):
    return ((wp2[0] - wp1[0])**2 + (wp2[1] - wp1[1])**2)**0.5

def path_length(wp_list):
    dist = 0
    for wp1, wp2 in zip(wp_list[:-1], wp_list[1:]):
        dist += segment_length(wp1, wp2)
    return dist

if __name__=='__main__':
    import matplotlib.pyplot as plt
    path = [[0, 0, 0], [5, 0, 0], [8, 6, 0], [1, 5, 0]]
    smooth = fermat(path)[0]
    wp_list, curvature, dist = path_grad(smooth)

    turn_velocity = []
    turn_dist = []
    cum_dist = np.cumsum(dist)
    for i in range(len(curvature)):
        a, b = turn_vel(curvature, i, dist)
        turn_velocity.append(a)
        turn_dist.append(b)

    turn_dist = cum_dist - np.array(turn_dist)
    p_length = path_length(smooth)
    # d = np.arange(0, , 0.01)
    dt = 0.01
    v = [0]
    p = [0]
    t = [0]
    a = [0]
    j = [0]
    i = 1
    c = 0
    max_jerk = LosSettings.max_acc/4

    mod = ReferenceModel(1, 1, LosSettings.max_acc, LosSettings.max_acc, dt)

    while p[i-1] < p_length:
        ok = False
        if p[i-1] > turn_dist[c]:
            if p[i-1] < cum_dist[c]:
                vel, acc, jerk = mod.update(turn_velocity[c])
            else:
                vel, acc, jerk = mod.update(LosSettings.cruise_speed)
                c += 1
        else:
            vel, acc, jerk = mod.update(LosSettings.cruise_speed)
        a.append(acc)
        v.append(vel)
        p.append(p[i-1] + v[-1]*dt + 0.5*a[-1]*dt**2)
        t.append(t[i-1]+dt)
        i+=1
        if t[-1] > 500:
            break


    # turn_velocity.insert(0, 0)

    plt.figure(1)
    plt.plot(p, v)
    plt.plot(cum_dist, curvature)
    plt.plot(cum_dist, turn_velocity)
    plt.plot(turn_dist, np.full(len(turn_dist), 0.4), '*')
    # plt.plot(cum_dist, np.full(len(turn_dist), 0.4), 'or')
    plt.plot(p, a)
    plt.legend(('v', 'curv', 'turn_vel', 'turn_dist'))
    # plt.plot(p, np.array(j)*10)
    # plt.legend(('v', 'curv', 'turn_vel', 'turn_dist', 'j*10'))

    # plt.figure(2)
    # plt.plot(t, p)
    plt.show()
    from scipy.io import savemat
    savemat('data.mat', {'p': p, 'v':v, 'a':a, 'smooth':smooth, 'dist':dist, 'cum_dist':cum_dist, 't':t, 'curvature':curvature})