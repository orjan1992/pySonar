from messages.moosMsgs import MoosMsgs
import logging
import threading
LOG_FILENAME = 'testMooseMsgs.out'
logging.basicConfig(filename=LOG_FILENAME,
                    level=logging.DEBUG,
                    filemode='w',)


def print_msg(msg):
    # print('X: {}\tY: {}\tR: {}'.format(msg.lat, msg.long, msg.rot))
    print('heading: {}\tlength: {}\n{}'.format(msg.bearing, msg.length, msg.bins))
    return True


def print_pos_msg(msg):
    print('X: {}\tY: {}\tR: {}'.format(msg.x, msg.y, msg.head))
    # print('heading: {}\tlength: {}\n{}'.format(msg.heading, msg.length, msg.bins))
    return True


moos = MoosMsgs()
moos.set_on_sonar_msg_callback(print_msg)
# moos.set_on_pos_msg_callback(print_pos_msg)
moos.run()


# t = threading.Thread(name='my_service', target=run_moos)
# t.start()
input('press enter')
# t.
# t.join()