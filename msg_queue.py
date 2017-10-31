import pymoos
import time

comms = pymoos.comms()
def on_connect():
	comms.register('simple_var',0.1)
	comms.register('test', 0.1)
	return True

def m(msg):
	print('msg1')
	msg.trace()
	return True
	
def m2(msg):
	print('msg2')
	msg.trace()
	return True
	
def main():
# my code h e r e
	comms.set_on_connect_callback( on_connect )
	comms.add_active_queue('the_queue', m)
	comms.add_message_route_to_active_queue('the_queue', 'simple_var')
	comms.add_active_queue('the_queue2', m2)
	comms.add_message_route_to_active_queue('the_queue2', 'test')
	comms.run('localhost' , 9000 , 'pymoos')
	i = 0
	while True:
		time.sleep(1)
		comms.notify('simple_var', 'hello world%i'%i, pymoos.time())
		comms.notify('test', 'held%i'%i, pymoos.time())
		i = i+1
		
		
if __name__ == "__main__":
	main()
