import pymoos
import time
from struct import unpack, calcsize
import codecs

comms = pymoos.comms()


def on_connect():
    comms.register('bins', 0.1)
    return True


def main():
    # my code h e r e
    print('test')
    comms.set_on_connect_callback(on_connect)
    comms.run('localhost', 9000, 'pymoos')
    time.sleep(1)
    a = comms.fetch()
    for msg in a:
        # test = msg.binary_data().decode('utf_8', 'replace')
        # print(msg.binary_data())
        data = msg.binary_data().encode('latin-1')
        tmp = unpack('>dH{:d}f'.format((len(data) - 10) // 4), data)
        head = tmp[0]
        length = tmp[1]
        bins = tmp[2:]
        print(bins)


if __name__ == "__main__":
    main()
