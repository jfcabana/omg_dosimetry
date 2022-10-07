# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 08:10:23 2018

@author: caje1277
"""
import calibration
import imageRGB
import numpy as np

LUT_file = 'S:\\Python\\EBT3\\ebt3_pylinac_module\\DemoCalibration\myLUT.pkl'
dose_file = 'S:\\Python\\EBT3\\ebt3_pylinac_module\\DemoCalibration\\NominalDoses\\Nominal_0_50_100_150_200_250_300_350.tif'

lut = calibration.load_lut(LUT_file)
img = imageRGB.load(dose_file)

#%%
for i in range(0,lut.npixel):
    profile = np.interp(lut.lat_pos[i], lut.profile[:,0], lut.profile[:,1]) / 100  
    img.array[i,:] = img.array[i,:].astype(float) * profile