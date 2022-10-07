# -*- coding: utf-8 -*-
"""
Wrapper script for the Gafchromic calibration module.

Usage: python CreateMultichannelLUT.py 'path to config.ini file'

Provide the path to a config.ini file as argument to automatically define
all required parameters. Calibration is performed and LUT is saved.

"""

#%% import librairies
import calibration
import os
import sys
import numpy as np
import configparser
import matplotlib.pyplot as plt
#config_file = sys.argv[1:]
config_file = 'S:\Python\EBT3\ebt3_pylinac_module\DemoCalibration\config.ini'

#%% Parse the parameters inside config file

def GetConfig(Config, section, option, default):
    dtype = type(default)   
    if not Config.has_section(section) or not Config.has_option(section,option):
        return default
    if dtype == str:
        return Config.get(section,option)
    elif dtype == float:
        return Config.getfloat(section,option)
    elif dtype == bool:
        return Config.getboolean(section,option)

Config = configparser.ConfigParser()
Config.read(config_file)
#Config.sections()
#Config.options('Beam')

img_folder = GetConfig(Config,'Path','img_folder','')
profile_file = GetConfig(Config,'Path','profile_file','')
output_name = GetConfig(Config,'Path','output_name', os.path.join(img_folder,'myLUT'))

doses = Config.get('Beam','doses')
doses = np.fromstring(doses,sep=',')
output = GetConfig(Config,'Beam','output',1.0)
profile_correction = GetConfig(Config,'Beam','profile_correction',True)

expose_date = GetConfig(Config,'Info','expose_date','')
expose_user = GetConfig(Config,'Info','expose_user','')
scan_date = GetConfig(Config,'Info','scan_date','')
scan_user = GetConfig(Config,'Info','scan_user', '')
scanner_id = GetConfig(Config,'Info','scanner_id','')

film_select = GetConfig(Config,'Film','film_select','Auto')
film_width = GetConfig(Config,'Film','film_width','Auto')
film_height = GetConfig(Config,'Film','film_height','Auto')
film_crop = GetConfig(Config,'Film','film_crop', 0.3)
        
#%% Compute LUT

print('Computing calibration on images found in folder {}'.format(img_folder))
myLUT = calibration.LUT(img_folder, doses=doses, output=output, profile_correction=profile_correction, profile_file=profile_file)
myLUT.expose_date = expose_date
myLUT.expose_user = expose_user 
myLUT.scan_date = scan_date
myLUT.scan_user = scan_user
myLUT.scanner_id = scanner_id

print('Calibration completed succesfully!')
#%% Save the LUT object
filename = output_name + '.pkl'
print('Saving lut file as {}'.format(filename))
calibration.save_lut(myLUT,filename)

#%% Save some graphics to output folder tp make sure everything went smoothly
f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8))
plt.close(f) 
myLUT.films.plot_roi(ax=ax1)
myLUT.films.plot_profile(ax=ax2)
myLUT.plot_calibration_curves(mode='both',ax=ax3)
f.tight_layout()
f.savefig(output_name + '.pdf', bbox_inches='tight', dpi=300)
f.savefig(output_name + '.png', bbox_inches='tight', dpi=300)
print('Results report saved as {}'.format(output_name + '.pdf'))