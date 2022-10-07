# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 08:57:22 2018

@author: caje1277
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import calibration

lut_file = 'S:\Python\EBT3\ebt3_pylinac_module\DemoCalibration\myLUT.pkl'
lut = calibration.load_lut(lut_file)

i = 400

# We must sort LUT by increasing gray value (decreasing dose) for fitting
if lut.profile_correction:
    p_lut = lut.lut[::-1,i,1:6]  
else:
    p_lut = lut.lut[::-1,1:6]        
                    
def rational_linear(x, a, b, c):
    return -c + b/(x-a)

def drational_linear(x, a, b, c):
    return -b/(x-a)**2
    
xdata = np.linspace(0.2,0.7,100)
ydata = p_lut[:,0]
xdata_m = p_lut[:,1]
xdata_r = p_lut[:,2]
xdata_g = p_lut[:,3]
xdata_b = p_lut[:,4]

p0 = [0.1, 50., 200.]
popt_m, pcov_m = curve_fit(rational_linear, xdata_m, ydata)
popt_r, pcov_m = curve_fit(rational_linear, xdata_r, ydata)
popt_g, pcov_m = curve_fit(rational_linear, xdata_g, ydata)
popt_b, pcov_m = curve_fit(rational_linear, xdata_b, ydata)


#%%
fig, (ax1,ax2) = plt.subplots(2,1)

ax1.plot(xdata_m, ydata, 'ko', label='data_m')
ax1.plot(xdata, rational_linear(xdata, *popt_m), 'k-')
ax1.plot(xdata_r, ydata, 'ro', label='data_m')
ax1.plot(xdata, rational_linear(xdata, *popt_r), 'r-')
ax1.plot(xdata_g, ydata, 'go', label='data_g')
ax1.plot(xdata, rational_linear(xdata, *popt_g), 'g-')
ax1.plot(xdata_b, ydata, 'bo', label='data_b')
ax1.plot(xdata, rational_linear(xdata, *popt_b), 'b-')

ax2.plot(xdata_m, drational_linear(xdata_m, *popt_m), 'ko', label='data_m')
ax2.plot(xdata, drational_linear(xdata, *popt_m), 'k-')
ax2.plot(xdata_r, drational_linear(xdata_r, *popt_r), 'ro', label='data_m')
ax2.plot(xdata, drational_linear(xdata, *popt_r), 'r-')
ax2.plot(xdata_g, drational_linear(xdata_g, *popt_g), 'go', label='data_g')
ax2.plot(xdata, drational_linear(xdata, *popt_g), 'g-')
ax2.plot(xdata_b, drational_linear(xdata_b, *popt_b), 'bo', label='data_b')
ax2.plot(xdata, drational_linear(xdata, *popt_b), 'b-')

plt.show()
