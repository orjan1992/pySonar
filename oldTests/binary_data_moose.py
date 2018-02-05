import pymoos
import time
#s i m p l e example which u s e s an a c t i v e queue t o h a n d l e r e c e i v e d m e s s a g e s
#you can send any number o f m e s s a g e s t o any number o f a c t i v e q ueues
#h e r_unit e we send b i n a r_unit y data j u s t t o show we can

comms = pymoos.comms()


def c():
    comms.register('binary_var', 0)
    return True


def queue_callback(msg):

    if msg.is_binary():
        b = bytearray(msg.binary_data())
        print('received ' + str(len(b)) + ' bytes')
    return True


def main():
    comms.set_on_connect_callback(c)
    comms.add_active_queue('the_queue', queue_callback)
    comms.add_message_route_to_active_queue('the_queue', 'binary_var')
    comms.run('localhost', 9000, 'pymoos')
    while True:
        time.sleep(1)
        x = bytearray([0, 3, 0x15, 2, 6, 0xAA])
        # print(type(x))
        # print(x)
        # print(type(bytes(x)))
        # print(bytes(x))
        comms.notify_binary('binary_var', bytes(x), pymoos.time())

if __name__=="__main__":
    main()