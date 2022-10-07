# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 13:37:08 2018

@author: caje1277
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets  import RectangleSelector

def test():
    xpos = []
    ypos = []
    
    xdata = np.linspace(0,9*np.pi, num=301)
    ydata = np.sin(xdata)
    
    fig, ax = plt.subplots()
    line, = ax.plot(xdata, ydata)
    
    
    def line_select_callback(eclick, erelease):
        ax = plt.gca()
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
    
        rect = plt.Rectangle( (min(x1,x2),min(y1,y2)), np.abs(x1-x2), np.abs(y1-y2), Fill=False )
        ax.add_patch(rect)
        
        xpos.append(x1)
        ypos.append(y1)
    
    
    rs = RectangleSelector(ax, line_select_callback,
                           drawtype='box', useblit=False, button=[1], 
                           minspanx=5, minspany=5, spancoords='pixels', 
                           interactive=False)
    
    plt.show()
    return rs

rs = test()