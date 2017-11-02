import csv
from itertools import islice

class ReadCsvFile(object):
    """
    Read CSV log files
    """

    nRows = 0

    def __init__(self, filename):
        self.file = open(filename, newline='')
        spamreader = csv.reader(self.file, delimiter=';')
        self.nRows = sum(1 for row in spamreader)

    def close(self):
        self.file.close()

    def readRows(self, start, n):
        self.file.seek(0)
        reader = csv.DictReader(self.file, delimiter=';', fieldnames=['time', 'ip', 'port', 'data'])
        for row in islice(reader, start, start+n):
            print(row['port'])
