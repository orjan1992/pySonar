import binascii
import csv
import struct
from itertools import islice
from math import pi
import numpy as np
from datetime import datetime, timedelta
from os import walk
import socket
import sys
from time import sleep


class ReadCsvFile(object):
    """
    Read CSV log files
    """
    SONAR_START = 64 # 0x40 = 64 = '@'
    LINE_FEED = 10 # 0x0A = 10 = 'LF'
    GRAD2RAD = pi / (16 * 200)
    curSonarMsg = bytearray()
    curSonarMsgTime = ''
    messagesReturned = 0

    def __init__(self, filename, sonarPort = 4002, posPort=13102, cont_reading=False):
        self.file = open(filename, newline='')
        self.reader = csv.DictReader(self.file, delimiter=';', fieldnames=['time', 'ip', 'port', 'data'])
        self.out_file = open('test.csv', 'w', newline='')
        self.writer = csv.writer(self.out_file, delimiter=',')
        self.cont_reading = cont_reading
        if cont_reading:
            tmp = filename.split('/')
            self.file_path = '/'.join(tmp[:-1])
            cur_file = tmp[-1]
            self.file_list = []
            for (dirpath, dirnames, filenames) in walk(self.file_path):
                self.file_list.extend(filenames)
                break
            self.file_list.sort()
            self.file_number = self.file_list.index(cur_file)
        self.sonarPort = sonarPort
        self.posPort = posPort


    def close(self):
        self.file.close()
        self.out_file.close()

    def readNextRow(self):
        """
        Read next msg
        :return: msg
        """
        return list(islice(self.reader, 0, 1))

    def read_next_msg(self):
        """
        Read next msg
        :return: msg
        """
        msg = self.readNextRow()
        try:
            if int(msg[0]['port']) == self.sonarPort:
                return self.sonarMsgToByteArray(msg[0]['data'])
            else:
                return 0
        except IndexError:
            if self.cont_reading:
                return self.start_again()
            else:
                return -1

    def start_again(self):
        try:
            self.file_number += 1
            self.file = open('/'.join([self.file_path, self.file_list[self.file_number]]), newline='')
            self.reader = csv.DictReader(self.file, delimiter=';', fieldnames=['time', 'ip', 'port', 'data'])
            return self.read_next_msg()
        except FileNotFoundError:
            return -1

    def sonarMsgToByteArray(self, byteArrayRow):
        bArray = bytearray.fromhex(byteArrayRow)
        return bArray


if __name__ == '__main__':
    if sys.argv.__len__() > 1:
        step = True
    else:
        step = False

    filename = 'UdpHubLog_4001_2017_11_02_09_00_03.csv'
    reader = ReadCsvFile(filename)
    TCP_IP = '127.0.0.1'
    source_port = 5555
    BUFFER_SIZE = 1024
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # s.bind((TCP_IP, Server_port))
    # print('Waiting for connection')
    # s.listen(1)
    # conn, addr = s.accept()
    # print('Connection address:', addr)
    # input("Press Enter to start sendind...")
    tmp = reader.read_next_msg()
    counter = 1
    while not(tmp == -1):
        if not(tmp == 0):
            s.sendto(tmp, (TCP_IP, source_port))
            counter += 1
            if step:
                input("Press Enter to continue...")
            else:
                sleep(0.01)
        tmp = reader.read_next_msg()
    s.close()
    print('Connection Closed, sent %d packages' % counter)
