# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 08:06:05 2018

@author: caje1277
"""

import imageRGB

file = "H:\\ExportRS\\5 ORL_CUSM ORL_5\\JF-v2-8\\JF-v2-8 23 Aug 2018, 07_51_23 (hr_min_sec)\\Test QA\\RD1.2.752.243.1.1.20180925151404825.4000.35285.3.256.2.dcm"
dose = imageRGB.load(file)
dose.plotCB()