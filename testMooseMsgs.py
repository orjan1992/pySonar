from messages.moosMsgs import MoosMsgs
import logging
import threading
LOG_FILENAME = 'logging_example.out'
logging.basicConfig(filename=LOG_FILENAME,
                    level=logging.DEBUG,
                    filemode='w',)


def print_msg(msg):
    # print('X: {}\tY: {}\tR: {}'.format(msg.lat, msg.long, msg.rot))
    print('heading: {}\tlength: {}\n{}'.format(msg.heading, msg.length, msg.bins))


moose = MoosMsgs()
moose.set_on_sonar_msg_callback(print_msg)


# t = threading.Thread(name='my_service', target=run_moos)
# t.start()
input('press enter')
# t.
# t.join()