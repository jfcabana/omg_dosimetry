import matplotlib.pyplot as plt
import numpy as np

def moved_and_pressed(event):
    if event.button==1:
        x = np.append(line.get_xdata(), event.xdata)
        y = np.append(line.get_ydata(), event.ydata)
        line.set_data(x, y)
        fig.canvas.draw()

fig, ax = plt.subplots(1,1, figsize=(5,3), dpi=100)
line, = ax.plot([], [], 'k')
ax.set_xlim(0,10); ax.set_ylim(0,10)
cid = fig.canvas.mpl_connect('motion_notify_event', moved_and_pressed)
plt.show()