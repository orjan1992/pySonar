from math import pi
from datetime import datetime, timedelta

def Wrap2pi(angle):
    return (angle + pi) % (2 * pi) - pi

def getTimeCsv(timeStr):
    # Maybe try except here
    tmp = timeStr.split(' ')
    date = tmp[0]
    time = tmp[1]
    date_tmp = date.split('.')
    time_tmp = time.split('.')
    date_time = datetime(int(date_tmp[0]), int(date_tmp[1]), int(date_tmp[2]), int(time_tmp[0]), int(time_tmp[1]), int(time_tmp[2]), int(time_tmp[3]))
    return date, time, date_time

def getTimeLog(time_in):
    date_time = datetime(1899, 12, 30) + timedelta(days=time_in)
    date = date_time.strftime('%Y.%m.%d')
    time = date_time.strftime('%H.%M.%S.%f')
    return date, time, date_time

def get_time_csv_file_name(fname):
    # Maybe try except here
    split = fname.split('_')[2:]
    split[-1] = split[-1].split('.')[0]
    tmp = list(map(int, split))
    return  datetime(year=tmp[0], month=tmp[1], day=tmp[2], hour=tmp[3], minute=tmp[4], second=tmp[5])
