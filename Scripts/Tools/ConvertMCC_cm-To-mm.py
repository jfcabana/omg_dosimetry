# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 10:02:50 2019

@author: caje1277
"""
import os
import csv

#%%
path_in = 'P:\\Projets\\CRIC\\Physique_Medicale\\VersaHD1\\srs\\ComparaisonDetecteur\\Films\\DoseFilm_spline\\MCC\\'
path_out = path_in + 'convert\\'
files = os.listdir(path_in)

for file in files:
    fileBase, fileext = os.path.splitext(file)   
    if fileext != '.mcc':
        continue
    
    fileIn = path_in + fileBase + '.mcc'
    fileOut = path_out + fileBase + '.mcc'
    
    # Read data
    data = []
    with open(fileIn, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for row in csv_reader:
            if len(row) > 0:
                if 'FIELD_INPLANE=' in row[-1]:
                    fs = row[-1].split('=')[-1]
                    row[-1] = 'FIELD_INPLANE=' + str(float(fs.replace(',','.'))*10)
                if 'FIELD_CROSSPLANE=' in row[-1]:
                    fs = row[-1].split('=')[-1]
                    row[-1] = 'FIELD_CROSSPLANE=' + str(float(fs.replace(',','.'))*10)
                data.append(row)
                
    # Write back
    with open(fileOut, 'w',  newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_MINIMAL)   
        for i in range(0, len(data)):
             csvwriter.writerow(data[i])