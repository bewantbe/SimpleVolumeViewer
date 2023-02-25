# multi-threading test
import time
import threading
import queue

class InterpThread(threading.Thread):
    def __init__(self, name, que):
        super().__init__(name = name)
        self.que = que
    
    def run(self):
        b = ' '
        while b:
            b = input(f'Thread {self.name}: Order? ')
            self.que.put(b)

if __name__ == '__main__':
    que_th = queue.Queue()
    t1 = InterpThread('haha', que_th)

    t1.start()
    while t1.is_alive():
        time.sleep(5.0)
        msg_id = 0
        while t1.is_alive() and not que_th.empty():
            msg_id += 1
            print(f'Main received: {msg_id}', que_th.get())
        else:
            print('Main: no message.')
