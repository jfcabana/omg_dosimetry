# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 12:09:07 2023

@author: caje1277
"""
import pickle
import bz2

filename="P:\\Projets\\CRIC\\Physique_Medicale\\Films\\OMG_git\\scripts\\demo\\files\\calibration\\Demo_calib.pkl"

try:
    file = bz2.open(filename, 'rb')
    lut = pickle.load(file)
except:
    file = open(filename, 'rb')
    lut = pickle.load(file)

file.close()

# try file = open(filename, 'rb'):
#     print("Hourra!")
# except:
#     print("Prout!")