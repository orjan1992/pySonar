from messages.moosMsgs import MoosMsgs
import logging
from ogrid.rawGrid import RawGrid
import threading
LOG_FILENAME = 'testMooseMsgs.out'
logging.basicConfig(filename=LOG_FILENAME,
                    level=logging.DEBUG,
                    filemode='w',)


class Test(object):

    def __init__(self):
        self.moos = MoosMsgs()
        self.moos.set_on_sonar_msg_callback(self.print_msg)
        # moos.set_on_pos_msg_callback(print_pos_msg)
        self.moos.run()
        self.ogrid_conditions = [0.1, 20, 15, 0.5]
        self.grid = RawGrid(self.ogrid_conditions[0], self.ogrid_conditions[1], self.ogrid_conditions[2],
                            self.ogrid_conditions[3])

    def print_msg(self, msg):
        a = self.grid.get_p()
        print('heading: {}\tlength: {}\n{}'.format(msg.bearing, msg.length, msg.data))
        return True


def print_pos_msg(msg):
    print('X: {}\tY: {}\tR: {}'.format(msg.x, msg.y, msg.head))
    # print('heading: {}\tlength: {}\n{}'.format(msg.heading, msg.length, msg.bins))
    return True

def main():
    a = Test()
    input('press enter')


if __name__ == "__main__":
    main()