from random import random
import threading
import time

result = None
result_available = threading.Event()

def background_calculation():

    # here goes some long calculation
    time.sleep(random() * 5 )

    # when the calculation is done, the result is stored in a global variable
    global result
    result = 42
    result_available.set()

    # do some more work before exiting the thread
    time.sleep(10)

def main():

    thread = threading.Thread(target=background_calculation)
    thread.start()

    # wait here for the result to be available before continuing
    print('before')
    result_available.wait()

    print('The result is', result)

if __name__ == '__main__':
    main()
    
    #%%
    
import numpy as np
import matplotlib.pyplot as plt

def onclick(event):
    global offset
    offset = event.ydata
    fig.canvas.mpl_disconnect(cid)
    plt.close()
    return

x = np.linspace(0, 2.0 * np.pi, 200)
y = np.sin(x)
fig = plt.figure()
plt.plot(x, y)
plt.title('Mouse left-click on the desired offset')
cid = fig.canvas.mpl_connect('button_press_event', onclick)
while not 'offset' in locals():
    plt.pause(5)
print('Offset =', offset)