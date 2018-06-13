from numpy import sign

class ReferenceModel:

    def __init__(self, w_n, zeta_n, vel_max, acc_max, dt):
        self.w_n = w_n
        self.w_n2 = w_n**2
        self.zeta_n = zeta_n
        self.zeta_w_2 = 2*zeta_n*w_n
        self.vel_max = vel_max
        self.acc_max = acc_max
        self.dt = dt

        self.low_pass = 0
        self.acc = 0
        self.vel = 0
        self.pos = 0

    def update(self, ref):
        self.low_pass += (ref - self.low_pass) * self.w_n * self.dt
        self.acc = self.sat(self.low_pass*self.w_n2 - self.w_n2*self.pos - self.vel*self.zeta_w_2, self.acc_max)

        self.vel += self.acc*self.dt
        self.pos += self.sat(self.vel, self.vel_max)*self.dt
        return self.pos, self.vel, self.acc

    @staticmethod
    def sat(val, max_val):
        if abs(val) > max_val:
            return sign(val) * max_val
        else:
            return val


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    dt = 0.1
    t = np.arange(0, 20, dt)
    x_d = np.ones(t.shape)*np.pi/2
    # x_d[t.shape[0]/2:] *= 2
    x = np.zeros(t.shape)
    v = np.zeros(t.shape)
    a = np.zeros(t.shape)
    ref = ReferenceModel(1, 1, 10 / np.pi, 0.1, dt)
    for i in range(0, np.shape(t)[0]):
        (x[i], v[i], a[i]) = ref.update(x_d[i])
    plt.plot(t, x_d, label='x_d')
    plt.plot(t, x, label='x')
    plt.plot(t, v, label='v')
    plt.plot(t, a, label='a')
    plt.legend()
    plt.grid()
    plt.show()
    # plt.plot(t, v)
    # plt.show()

