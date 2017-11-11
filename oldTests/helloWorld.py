import pymoos
import time

comms = pymoos.comms()


def on_connect():
    comms.register('test', 0.1)
    comms.register('simple_var', 0.1)
    comms.register('DB_TIME', 0.1)
    return


def main():
    # my code h e r e
    print('test')
    comms.set_on_connect_callback(on_connect)
    comms.run('localhost', 9000, 'pymoos')
    i = 0
    while True:
        time.sleep(1)
        comms.notify('simple_var', 'hello world%i' % i, pymoos.time())
        comms.notify('test', 'test2', pymoos.time())
        # map(lambda msg: msg.Trace(), comms.fetch())
        a = comms.fetch()
        if a.__len__() > 0:
            a.__getitem__(0).trace()


if __name__ == "__main__":
    main()
