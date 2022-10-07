# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 12:30:07 2019

@author: caje1277
"""
import pydicom
path = 'P:\Projets\CRIC\Physique_Medicale\Films\QA Patients\\123456789\ThLSD_1A\DoseRS\\123456789_ThLSD1A_RS.dcm'
#self._sid = sid
#self._dpi = dpi
# read the file once to get just the DICOM metadata
metadata = pydicom.read_file(path, force=True)
_original_dtype = metadata.pixel_array.dtype
# read a second time to get pixel data
if isinstance(path, BytesIO):
    path.seek(0)
ds = pydicom.read_file(path, force=True)

# Compute dose scaling factor
if doseScalingFactor is None:
    doseScalingFactor = self.doseScalingFactor
if doseUnits.lower() == 'cgy' and self.metadata.DoseUnits.lower() == 'gy':
    doseScalingFactor = doseScalingFactor*100
    self.metadata.DoseUnits = 'cGy'      
if doseUnits.lower() == 'gy' and self.metadata.DoseUnits.lower() == 'cgy':
    doseScalingFactor = doseScalingFactor/100
    self.metadata.DoseUnits = 'Gy'
self.metadata.DoseGridScaling = doseScalingFactor

if dtype is not None:
    self.array = float(doseScalingFactor)*ds.pixel_array.astype(dtype)
else:
    self.array = float(doseScalingFactor)*ds.pixel_array
    
# convert values to proper HU: real_values = slope * raw + intercept    
if self.metadata.SOPClassUID.name == 'CT Image Storage':
    self.array = int(self.metadata.RescaleSlope)*self.array + int(self.metadata.RescaleIntercept)