# -*- coding: utf-8 -*-
""" OMG_Tiff2Analysis.py
    - More Details to Come
"""
__author__ = "Peter Truong"
__contact__ = "petertruong.cissso@ssss.gouv.qc.ca"
__version__ = "19 février 2024"

from omg_dosimetry import analysis, tiff2dose
import os, sys, ctypes, pickle
import tkinter as tk
from tkinter import filedialog
root = tk.Tk()                              # Declare root (top-level instance)
root.withdraw()                             # Show only dialog without any other GUI elements by hiding root window
root.attributes("-topmost", True)           # Top-level window display priority
import pydicom
import matplotlib.pyplot as plt
plt.ion()           # Interactive Mode: ON

### Parameter Initialization
info = dict(author = "PT", 
            unit = "CL4", 
            film_lot = "EBT-3 C2",
            scanner_id = "Epson 10000XL", 
            date_exposed = "2024-02-06",
            date_scanned = "2024-02-07",
            wait_time = "24h", 
            notes = "Test Gamma with Time")

### Look-up Table (LUT) Path Initialization
landscape = False
LatCor = True
if landscape: # Landscape Orientation
    lut_file = (r"\\SVWCT2Out0455\Phys\Répertoires communs\Radiotherapie Externe\Film_QA\Calibration_LUT"
                r"\2022-10-04\C2_3Gy_72dpi_landscape.pkl")
else: # Portrait Orientation    
    if LatCor: lut_file = (r"\\SVWCT2Out0455\Phys\Répertoires communs\Radiotherapie Externe\Film_QA\Calibration_LUT"
                            r"\2023-09-12 (C2 LatCor)\C2_3Gy_LUT_LatCor_9MeV_2023-09-12.pkl")
    else: lut_file = (r"\\SVWCT2Out0455\Phys\Répertoires communs\Radiotherapie Externe\Film_QA\Calibration_LUT"
                      r"\2023-09-12 (C2 LatCor)\Sans LatCor\C2_3Gy_LUT_9MeV_2023-09-12.pkl")
    

### Tiff2Dose Parameters
tiff_2_dose, tiff_2_dose_show_pdf = 1, 0
clip = 600
if landscape: rot_scan = 1
else: rot_scan = 0
normFilm_selection = False

### Tiff2Analysis Parameters
dose_2_analysis, dose_2_analysis_show_pdf = 1, 0
analysis_publish_pdf = False
pickle_save = False
crop_film = 1
flipLR, flipUD = 1, 0
rot90 = 0

shift_x, shift_y = 0, 0
markers_center = None
#markers_center = [0, 0, 45]         # Based on DICOM coordinates in mm (LR, IS, AP)

#normalisation = 1.00
#normalisation = "ref_roi"
normalisation = "norm_film"
norm_film_MU = 300

### Normalization Reference (Eclipse)
norm_film_ref_MU = 300
norm_film_ref_dose = 315.5          # qaphys_dosimetrie_filmGaf/Calibration/C2 (average diagonal profile in Eclipse)
norm_film_dose = norm_film_MU / norm_film_ref_MU * norm_film_ref_dose

### Gamma Analysis Parameters
threshold = 0.10
norm_val = "max"
film_filt = 3

### Path Initialization
path_base = r"\\SVWCT2Out0455\Phys\Répertoires communs\Radiotherapie Externe\Film_QA"

### Script Functionality
def main():
    
    ### Scanned Film Image Selection
    path_scan = filedialog.askopenfilename(parent = root, title = "Select Scanned Film to Convert To Dose", 
                                           initialdir = path_base)
    if not path_scan:
        print("No image selected. Exiting Script... ")
        sys.exit()
    scan_name = os.path.basename(os.path.splitext(path_scan)[0])
    
    ### Scanned Normalization Film Image Selection
    if normFilm_selection:
        response = ctypes.windll.user32.MessageBoxW(0, "Was the normalization film scanned separate? If so, would you " \
                                                    "wish to proceed with selecting the normalization film scan?", 
                                                    "Normalization Film Selection", 0x1000 | 0x20 | 0x3)
        if response == 6: 
            path_normFilm = filedialog.askopenfilename(parent = root, title = "Select Scanned Normalization Film " \
                                                       "to Convert To Dose")
            normFilm_name = os.path.basename(os.path.splitext(path_normFilm)[0])
        elif response == 2: 
            print("Cancel option was selected. Exiting Script..." )
            sys.exit()
        else: path_normFilm = None
    else: path_normFilm = None
    
    ### Reference Eclipse Dose Plan File Selection
    path_doseEclipse = filedialog.askopenfilename(parent = root, title = "Select Reference Eclipse Dose Plan File", 
                                                  filetypes = [("Eclipse Dose Plane DICOM File", ".dcm")], 
                                                  initialdir = os.path.dirname(os.path.dirname(path_scan)))
    if not path_doseEclipse:
        print("No image selected. Exiting Script... ")
        sys.exit()
    
    ### DICOM Information Extract
    try: 
        ds = pydicom.dcmread(path_doseEclipse)
        patient_ID, plan_ID = ds.PatientID, ds.DoseComment
    except:
        print("Invalid DICOM file selected. ")
        sys.exit()
    
    ### Initialize Folder Structure
#    path_plan = os.path.join(path_base, patient_ID, plan_ID)
    path_plan = os.path.dirname(os.path.dirname(path_doseEclipse))
    path_doseFilm, path_analyse = os.path.join(path_plan, "DoseFilm"), os.path.join(path_plan, "Analyse")
    
    ### Create Folders
#    if not (os.path.exists(path_plan)): os.makedirs(path_plan)             # Should exist if taken from os.path.dirname
    if not os.path.exists(path_doseFilm): os.makedirs(path_doseFilm)     
    if not os.path.exists(path_analyse): os.makedirs(path_analyse)  
    
    ### Create Tiff2Dose
    if tiff_2_dose:
        gaf = tiff2dose.Gaf(path = path_scan, lut_file = lut_file, info = info, clip = clip, rot90 = rot_scan)
        gaf_dose_tif = os.path.join(path_doseFilm, scan_name) + ".tif"
        gaf.dose_opt.save(gaf_dose_tif)
        gaf.publish_pdf(gaf_dose_tif[:-4] + ".pdf", open_file = tiff_2_dose_show_pdf)
        if pickle_save: pickle.dump(gaf, open(gaf_dose_tif[:-4] + ".pkl", "wb"))
        if path_normFilm:
            gaf_norm = tiff2dose.Gaf(path = path_normFilm, lut_file = lut_file, info = info, clip = clip, 
                                     rot90 = rot_scan)
            gaf_norm_dose_tif = os.path.join(path_doseFilm, normFilm_name) + ".tif"
            gaf_norm.dose_opt.save(gaf_norm_dose_tif)
            gaf_norm.publish_pdf(gaf_norm_dose_tif[:-4] + ".pdf", open_file = tiff_2_dose_show_pdf)
            if pickle_save: pickle.dump(gaf_norm, open(gaf_norm_dose_tif[:-4] + ".pkl", "wb"))
        
    ### Analyze
    if dose_2_analysis:
        if path_normFilm:
            film = analysis.DoseAnalysis(film_dose = gaf_dose_tif, ref_dose = path_doseEclipse, 
                                        norm_film_dose = gaf_norm_dose_tif, flipLR = flipLR, flipUD = flipUD, 
                                        rot90 = rot90)
        else:
            film = analysis.DoseAnalysis(film_dose = gaf_dose_tif, ref_dose = path_doseEclipse, flipLR = flipLR, 
                                        flipUD = flipUD, rot90 = rot90)
        
        if normalisation == "norm_film": film.apply_factor_from_roi(norm_dose = norm_film_dose)
        if crop_film: film.crop_film()
        film.register(shift_x = shift_x, shift_y = shift_y, threshold = 10, register_using_gradient = True, 
                      markers_center = markers_center)
        if normalisation == "ref_roi": film.apply_factor_from_roi()             # Normalize based on roi dose
        
        gamma_analysis(film, scan_name, path_analyse, doseTA = 3, distTA = 3, show_results = True, 
                      threshold = threshold, norm_val = norm_val, film_filt = film_filt, pickle_save = pickle_save)
        gamma_analysis(film, scan_name, path_analyse, doseTA = 2, distTA = 2, show_results = True, 
                      threshold = threshold, norm_val = norm_val, film_filt = film_filt, pickle_save = pickle_save)
        gamma_analysis(film, scan_name, path_analyse, doseTA = 1, distTA = 1, show_results = True,
                      threshold = threshold, norm_val = norm_val, film_filt = film_filt, pickle_save = pickle_save)
        return
        
def gamma_analysis(dose2analysis, filebase, path_analyse, doseTA = 3, distTA = 3, show_results = True, 
                  threshold = 0.10, norm_val = "max", film_filt = 3, pickle_save = True):
    filename = "{}_Facteur{:.2f}_Filtre{}_Gamma{}%-{}mm_report.pdf".format(filebase, 
                dose2analysis.film_dose_factor, film_filt, doseTA, distTA)
    fileout = os.path.join(path_analyse, filename)
    print("\nAnalyse en cours...")
    dose2analysis.gamma_analysis(doseTA = doseTA, distTA = distTA, threshold = threshold, norm_val = norm_val, 
                          film_filt = film_filt, local_gamma = False)
    print("")
    print("======================= Gamma {}/{} ============================".format(doseTA, distTA))
    print("      Gammma {}%/{}mm: Taux de passage={:.2f}%; Moyenne={:.2f}".format(doseTA, distTA, 
          dose2analysis.GammaMap.passRate, dose2analysis.GammaMap.mean))   
    print("==============================================================")
    if show_results: dose2analysis.show_results()
    if analysis_publish_pdf: 
        dose2analysis.publish_pdf(fileout, open_file = dose_2_analysis_show_pdf, show_hist = True, 
                                  show_pass_hist = True, show_varDistTA = False, show_varDoseTA = False, 
                                  x = None, y = None)
    if pickle_save: pickle.dump(dose2analysis, open(fileout[:-4] + ".pkl", "wb"))
    
def load_pickle(file_path = None, show_results = True):
    if not file_path: file_path = filedialog.askopenfilename(parent = root, filetypes = [("pickle", ".pkl")])
        
    pkl = pickle.load(open(file_path, "rb"))
    if show_results: pkl.show_results()
    
    return pkl
    
if __name__ == "__main__":
    main()