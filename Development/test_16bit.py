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
import numpy as np
import imageRGB


file = 'P:\\Projets\\CRIC\\Physique_Medicale\\Films\\Baseline scanner\\Transmission\\Transmission_001.tif'

im0 = pImage.open(file)
im0.save('S:/Python/EBT3/ebt3_pylinac_module/DemoCalibration/im0.tif')
# return an non-16 bit rgb image

im1 = imageio.imread(file)
plt.imshow(im1 / 65535. )
imageio.imsave('S:/Python/EBT3/ebt3_pylinac_module/DemoCalibration/im1.tif', im1)
# this works

im2 = tiff.imread(file)
tiff.imshow(im2)
tiff.imsave('S:/Python/EBT3/ebt3_pylinac_module/DemoCalibration/im2.tif', im2)
# this works


#%% Save metadata

from __future__ import print_function, unicode_literals

import json
import numpy
import tifffile  # http://www.lfd.uci.edu/~gohlke/code/tifffile.py.html

data = numpy.arange(256).reshape((16, 16)).astype('u1')
metadata = dict(microscope='george', shape=data.shape, dtype=data.dtype.str)
print(data.shape, data.dtype, metadata['microscope'])

metadata = json.dumps(metadata)
tifffile.imsave('microscope.tif', data, description=metadata)

with tifffile.TiffFile('microscope.tif') as tif:
    data = tif.asarray()
    metadata = tif[0].image_description
metadata = json.loads(metadata.decode('utf-8'))
print(data.shape, data.dtype, metadata['microscope'])