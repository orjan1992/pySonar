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
        log_type_a, log_type_b, date = (fname_split[0]).split('_')
        if fname_split[1] == 'npz' and log_type_a == 'scan' and log_type_b == 'lines':
            t = datetime.datetime.strptime(date, "%Y%m%d-%H%M%S")
            if t < end_time and t > start_time:
                prop_cycle = plt.rcParams['axes.prop_cycle']
                colors = prop_cycle.by_key()['color']

                scan_lines = np.load(file)['scan_lines']
                for i in range(np.shape(scan_lines)[0]):
                    plt.figure(1)
                    plt.plot(scan_lines[i, :])
                    plt.figure(2)
                    plt.plot(np.gradient(scan_lines[i, :]))
                plt.show()

    except:
        pass
    if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
        break
