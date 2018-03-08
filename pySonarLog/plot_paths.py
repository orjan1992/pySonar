import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import select
import sys

flist = os.listdir()
start_time = datetime.datetime(2017, 3, 6, 1, 31, 0)
end_time = datetime.datetime(2019, 1, 1, 0, 0, 0)
for file in flist:
    fname_split = file.split('.')
    try:
        log_type, date = (fname_split[0]).split('_')
        if fname_split[1] == 'npz' and log_type == 'paths':
            t = datetime.datetime.strptime(date, "%Y%m%d-%H%M%S")
            if t < end_time and t > start_time:
                prop_cycle = plt.rcParams['axes.prop_cycle']
                colors = prop_cycle.by_key()['color']

                tmp = np.load(file)
                paths = tmp['paths']
                pos = tmp['pos']


                color_index = 0
                for p in paths:
                    if p is not None:
                        p_array = np.array(p)
                        plt.plot(p_array[:, 1], p_array[:, 0], '--', color=colors[color_index])
                        plt.plot(p_array[0, 1], p_array[0, 0], '*', color=colors[color_index])
                        color_index += 1
                        if color_index == len(colors):
                            color_index = 0
                for i in range(np.shape(pos)[0]):
                    if np.any(pos[i, :] != 0):
                        start = i
                        break
                plt.plot(pos[start:, 1], pos[start:, 0], 'r')
                plt.show()

    except:
        pass
    if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
        break
