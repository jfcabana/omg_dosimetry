9# -*- coding: utf-8 -*-
"""
Script de conversion des LUT
"""
import os
import sys
import pickle
from omg_dosimetry import calibration
from omg_dosimetry import imageRGB

path_in = "P:\\Projets\\CRIC\\Physique_Medicale\\Films\\Calibrations\\"
path_out = "P:\\Projets\\CRIC\\Physique_Medicale\\Films\\Calibrations\\Convert\\"
# lut_file = "C14_calib_18h_trans_300ppp_0-9Gy.pkl"
files = os.listdir(path_in)
for file in files:
    if file.endswith(".pkl"):
        file_in = path_in + file
        file_out = path_out + file
        
        lut = calibration.load_lut(file_in)
        calibration.save_lut(lut, file_out)

        # sys.modules['calibration'] = calibration
        # sys.modules['imageRGB'] = imageRGB
        
        # with open(file_in, 'rb') as input:
        #     lut = pickle.load(input)
            
        # del sys.modules['calibration']
        # del sys.modules['imageRGB']
        
        # with open(file_out, 'wb') as output:
        #     pickle.dump(lut, output, pickle.HIGHEST_PROTOCOL)