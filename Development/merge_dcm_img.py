# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 16:45:40 2019

@author: caje1277
"""

import imageRGB
import os
import numpy as np

path = 'H:\\ExportRS\\VMAT_SEIN 0DOSI DOSI_STB_SEING\\SEIN_1A\\'

files = os.listdir(path)
img_list = []

for file in files: 
    img_file = os.path.join(path, file)
    filebase, fileext = os.path.splitext(file)
    
    if file == 'Thumbs.db':
        continue
    if os.path.isdir(img_file):
        continue
    
    img_list.append(imageRGB.load(img_file))
    
img = img_list[0]
new_array = np.stack(tuple(img.array for img in img_list), axis=-1)
img.array = np.sum(new_array, axis=-1)