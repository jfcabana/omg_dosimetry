# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 07:41:12 2023

@author: caje1277
"""
from PIL import Image

file_in = "P:\\Projets\\CRIC\\Physique_Medicale\\Films\\OMG_git\\scripts\\demo\\files\\tiff2dose\\scan\\A1A_Multi_6cm_001.tif"
file_out = "P:\\Projets\\CRIC\\Physique_Medicale\\Films\\OMG_git\\scripts\\demo\\files\\tiff2dose\\scan\\A1A_Multi_6cm_001_lzw.tif"

img = Image.open(file_in)
img.save(file_out, compression='lzw')

img_deflate = Image.open(file_out)