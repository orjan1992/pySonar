import numpy as np
from settings import GridSettings
from scipy.io import savemat, loadmat
def get_hit_inds(data, threshold):
        # Smooth graph
        smooth = np.convolve(data, np.full(GridSettings.smoothing_factor, 1.0/GridSettings.smoothing_factor), mode='full')
        data_len = len(smooth)
        s1 = smooth[:-2]
        s2 = smooth[1:-1]
        s3 = smooth[2:]

        # Find inital peaks and valleys
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

        # Remove consecutive peaks or valleys
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

        # Remove peaks and valleys with primary factor lower than 5
        mask = np.logical_and(smooth[peaks] - smooth[valleys[:-1]] > 5, smooth[peaks] - smooth[valleys[1:]] > 5)
        peaks = (np.array(peaks)[mask]).tolist()
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
                    if smooth[i] < smooth[i_sgn]:
                        valleys.remove(i_sgn)
                        i_sgn = i
                    else:
                        valleys.remove(i)
                else:
                    sgn = -1
                    i_sgn = i

        # Return peaks with a primary factor higher than threshold and transform to 800 bin length
        smooth_peaks = smooth[peaks]
        smooth_valleys = smooth[valleys]
        mask = np.logical_or(smooth_peaks - smooth_valleys[:-1] > threshold,
                             smooth_peaks - smooth_valleys[1:] > threshold)
        return np.array(peaks)[mask],peaks, valleys, smooth

if __name__=='__main__':
    tmp = loadmat('scanlines')
    data = tmp['scanlines']
    ad_low = tmp['ad_low']
    ad_span = tmp['ad_span']
    bearing = tmp['bearing']
    range_scale = tmp['range_scale']
    start = 0
    stop = len(range_scale[0])
    final_peaks = []
    smooth_peaks = []
    smooth_valleys = []
    smooth = []
    mask = np.ones(stop, dtype=bool)
    for i in range(start, stop):
        if len(data[0][i][0]) < 100:
            mask[i] = False
            continue
        final_peaks_, smooth_peaks_, smooth_valleys_, smooth_ = get_hit_inds(data[0][i][0], 50)

        final_peaks.append(final_peaks_)
        smooth_peaks.append(smooth_peaks_)
        smooth_valleys.append(smooth_valleys_)
        smooth.append(smooth_)
    savemat('scanlines_new.mat', {'scanlines': data[0][mask], 'ad_low': ad_low[0][mask],
                              'ad_span': ad_span[0][mask], 'bearing': bearing[0][mask],
                              'range_scale': range_scale[0][mask], 'final_peaks': np.array(final_peaks),
                              'smooth_peaks': np.array(smooth_peaks), 'smooth_valleys': np.array(smooth_valleys),
                              'smooth': np.array(smooth)})