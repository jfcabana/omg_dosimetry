# -*- coding: utf-8 -*-
"""
Created on Tue May 29 07:26:34 2018

@author: caje1277
"""
import matplotlib.pyplot as plt
import imageio
import tifffile as tiff
from PIL import Image as pImage
import cv2

file = 'S:/Python/EBT3/ebt3_pylinac_module/DemoCalibration/Calib8-14_001.tif'

im0 = pImage.open(file)
im0.save('S:/Python/EBT3/ebt3_pylinac_module/DemoCalibration/im0.tif')

im1 = imageio.imread(file)
imageio.imsave('S:/Python/EBT3/ebt3_pylinac_module/DemoCalibration/im1.tif', im1)

im2 = tiff.imread(file)
tiff.imsave('S:/Python/EBT3/ebt3_pylinac_module/DemoCalibration/im2.tif', im2)

im3 = cv2.imread(file, -1)
cv2.imwrite('S:/Python/EBT3/ebt3_pylinac_module/DemoCalibration/im3.tif', im3)
