import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def find_extremes(smooth):
    s1 = smooth[:-2]
    s2 = smooth[1:-1]
    s3 = smooth[2:]

    peaks = (np.array(
        np.nonzero(np.logical_or(np.logical_and(s1 < s2, s2 > s3), np.logical_and(s1 < s2, s2 == s3)))).reshape(
        -1) + 1).tolist()
    valleys = (np.array(
        np.nonzero(np.logical_or(np.logical_and(s1 > s2, s2 < s3), np.logical_and(s1 > s2, s2 == s3)))).reshape(
        -1) + 1).tolist()
    if peaks[0] != 0 and peaks[0] < valleys[0]:
        valleys.insert(0, 0)
    if peaks[-1] != data_len - 1 and peaks[-1] > valleys[-1]:
        valleys.append(data_len - 1)

    signed_array = np.zeros(data_len, dtype=np.int8)
    signed_array[peaks] = 1
    signed_array[valleys] = -1
    sgn = signed_array[0]
    i_sgn = 0
    for i in range(1, data_len):
        if signed_array[i] == 1:
            if sgn == signed_array[i]:
                peaks.remove(i_sgn)
            else:
                sgn = 1
            i_sgn = i
        elif signed_array[i] == -1:
            if sgn == signed_array[i]:
                valleys.remove(i_sgn)
            else:
                sgn = -1
            i_sgn = i
    return peaks, valleys

thresh = 50
scan = np.load('scanline2.npz')['scanline']
scan[100:133] = 90
smooth = np.convolve(scan, np.ones((10,))/10, mode='full')

# smooth = np.array([0, 10, 10, 20, 10, 25, 25, 25, 30, 5, 5, 10])
peak_list = []
valley_list = []
data_len = len(smooth)
i = 0

if smooth[0] > smooth[1]:
    peak_list.append(0)
else:
    valley_list.append(0)

for i in range(1, data_len-1):
    if smooth[i - 1] < smooth[i] > smooth[i + 1]:
        peak_list.append(i)
        continue
    if smooth[i - 1] > smooth[i] < smooth[i + 1]:
        valley_list.append(i)
        continue
    if smooth[i - 1] > smooth[i] == smooth[i + 1]:
        valley_list.append(i)
        continue
    if smooth[i - 1] < smooth[i] == smooth[i + 1]:
        peak_list.append(i)
        continue

if peak_list[-1] > valley_list[-1]:
    valley_list.append(data_len-1)

peak = np.array(peak_list, dtype=int)
valley = np.array(valley_list, dtype=int)

for i in range(1000):
    peak2, valley2 = find_extremes(smooth)

print(np.all(peak==peak2))
print(np.all(valley==valley2))

plt.plot(smooth)
plt.scatter(peak2, smooth[peak2], color='orange')
plt.scatter(valley2, smooth[valley2], color='green', marker='x')
plt.show()