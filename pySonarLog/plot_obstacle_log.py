import numpy as np
import cv2
import os
import time
import datetime

flist = os.listdir()
start_time = datetime.datetime(2018, 3, 5, 16, 8, 0)
end_time = datetime.datetime(2018, 3, 5, 16, 9, 30)
for file in flist:
    fname_split = file.split('.')
    try:
        if fname_split[1] == 'npz':
            t = datetime.datetime.strptime(fname_split[0], "%Y%m%d-%H%M%S")
            if t < end_time and t > start_time:
                im = np.load(file)['im']
                cv2.imshow(file, im)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    except:
        pass