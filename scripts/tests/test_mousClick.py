# -*- coding: utf-8 -*-
"""
Created on Fri May 12 12:02:13 2023

@author: caje1277
"""

import numpy as np
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt


def onclick(event, ax):
    # Only clicks inside this axis are valid.
    try: # use try/except in case we are not using Qt backend
        zooming_panning = ( fig.canvas.cursor().shape() != 0 ) # 0 is the arrow, which means we are not zooming or panning.
    except:
        zooming_panning = False
    if zooming_panning: 
        print("Zooming or panning")
        return
    if event.inaxes == ax:
        if event.button == 1:
            print(event.xdata, event.ydata)
            # Draw the click just made
            ax.scatter(event.xdata, event.ydata)
            ax.figure.canvas.draw()
        elif event.button == 2:
            # Do nothing
            print("scroll click")
        elif event.button == 3:
            # Do nothing
            print("right click")
        else:
            pass


fig, (ax1, ax2) = plt.subplots(1, 2)
# Plot some random scatter data
ax2.scatter(np.random.uniform(0., 10., 10), np.random.uniform(0., 10., 10))

fig.canvas.mpl_connect(
    'button_press_event', lambda event: onclick(event, ax2))
plt.show()


#%%
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

arr = np.arange(100, dtype=float).reshape(10, 10)
arr[~(arr % 7).astype(bool)] = np.nan
arr = np.ma.array (arr, mask=np.isnan(arr))
current_cmap = matplotlib.cm.get_cmap()
# current_cmap = matplotlib.colormaps.get_cmap()
current_cmap.set_bad(color='red')
plt.imshow(arr)


#%%
masked_array = np.ma.array (arr, mask=np.isnan(arr))
cmap = matplotlib.cm.jet
cmap.set_bad('k',1.)
plt.imshow(masked_array, interpolation='nearest', cmap=cmap)

#%%
import numpy as np
import matplotlib.pyplot as plt

f = plt.figure()
ax = f.add_subplot(111)
a = [
      [1, 3, 5, np.nan, 8, 9, np.nan],
      [11, 13, 51, 71, 18, 19, 10],
      [11, 31, 51, 71, 81, 91, 10],
      [10, 30, 50, 70, np.nan, np.nan, np.nan],
      [np.nan, 3, 5, np.nan, 8, 9, np.nan]
   ]
ax.imshow(a, interpolation='nearest', vmin=0, vmax=24)
f.canvas.draw()
plt.show()