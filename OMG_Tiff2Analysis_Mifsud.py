import analysis
import os
import tiff2dose
import matplotlib.pyplot as plt
plt.ion()
if __name__ == '__main__':

#%% ################################## Entrer vos informations ci-dessous ##################################
    #### Set film metadata ####
    info = dict(author = 'JFC',
                unit = 'Salle 3',
                film_lot = 'C7',
                scanner_id = 'Epson 12000XL',
                date_exposed = '2019/06/28',
                date_scanned = '2019/06/29',
                wait_time = '60 hr',
                notes = 'Scan en transmission avec la vitre.'
               )
    
    #### Set filenames and paths ####
    ID_patient = '20190909'
    ID_plan = 'Sein1A'
    
    #### Choose steps to perform ####
    tiff_2_dose = 1      # True(1) to convert the tiff film file to a dose file, False(0) to not do it
    tiff_2_dose_show_pdf = 1
    dose_2_analysis = 1   # True(1) to perform the gamma analysis, False(0) to not do it
    dose_2_analysis_show_pdf = False
    
    #### Scan2Dose parameters ####
    clip = 2000             # Entrer une valeur en cGy maximum où couper la dose. Si 'None', n'applique pas de threshold.
    rot_scan = 0            # Au besoin, mettre à 1 pour appliquer une rotation de 90 degrés sur l'image (si l'image de scan est verticale)
    
    # Choisir le bon fichier de calibration en décommentant seulement la bonne ligne
    #lut_file = 'P:\Projets\CRIC\Physique_Medicale\Films\Calibrations\C5_calib_24h_trans_LatCorr_0-500.pkl'
    lut_file = 'P:\Projets\CRIC\Physique_Medicale\Films\Calibrations\C5_calib_24h_trans_LatCorr_0-500_vitre.pkl'
    #lut_file = 'P:\Projets\CRIC\Physique_Medicale\Films\Calibrations\C6-XD_calib_24h_trans_LatCorr_0-24Gy.pkl'
    #lut_file = 'P:\Projets\CRIC\Physique_Medicale\Films\Calibrations\C6-XD_calib_24h_trans_LatCorr_0-24Gy_vitre.pkl'
    
    
    #### Dose2Analysis parameters ####
    crop_film = 1           # Mettre à 1 pour cropper l'image du film avant l'analyse, 0 si non
    flipLR = 0              # 1 if the film needs to be flipped in the left-right direction, 0 if not
    flipUD = 0              # 1 if the film needs to be flipped in the up-down direction, 0 if not
    rot90 = 2              # Number of 90 degrees rotation to apply to film dose
    shift_x = 0             # Shift to apply to the ref dose in the x direction (mm)
    shift_y = 0             # Shift to apply to the ref dose in the y direction (mm)
    markers_center = [96.2, -599.3, 211.6] # Coordinates of the makers intersection in mm (LR, IS, AP), as given in RayStation. = [-0.3, -722.7, 211] pour fantôme IMRT classique.
#    markers_center = [0.5, -38.0, 214.5]  # Coordonnées pour Shane
    
    # Choisir une des 3 méthodes de normalisation suivantes (commenter les 2 autres):
 #   normalisation = 3.333             # Si un chiffre, applique ce facteur de normalisation. 
    normalisation = 'ref_roi'       # Si 'ref_roi': sélectionne une ROI sur le film et normalisation par rapport à la dose de référence
#    normalisation = 'norm_film'     # Si 'norm_film': sélectionne le film de normalisation pour calculer le facteur par rapport à une dose attendue
    norm_film_MU = 1000             # Combien de MU délivrés pour le film de normalisation dans le setup standard                                       
    
    # Gamma analysis parameters
    # Le code plus bas fait l'analyse Gamma 3/3 et 2/2. On peut changer les paramètres suivants au besoin:
    threshold = 0.1         # Seuil de basses doses (0.1 = ne considère pas les doses < 10% du max de la dose ref)
    norm_val = 'max'        # 'max' pour normaliser par rapport à la dose maximum, ou entre une dose absolue en cGy pour normaliser sur une autre valeur
    film_filt = 3           # Taille du kernel de filtre à appliquer au film pour réduire le bruit. 3 est une bonne valeur, ne devrait être changé que si ne donne pas des résultats satisfaisants.

#%% ################################## NE RIEN CHANGER PLUS BAS ########################################

    #%% ################################### Set paths and file names ########################################
    
    path_base = 'P:\Projets\CRIC\Physique_Medicale\Films\QA Patients'
    path_patient = os.path.join(path_base, ID_patient)
    path_plan = os.path.join(path_patient, ID_plan)
    path_scan = os.path.join(path_plan, 'Scan')
    path_doseFilm = os.path.join(path_plan, 'DoseFilm')
    path_doseRS = os.path.join(path_plan, 'DoseRS')
    path_analyse = os.path.join(path_plan, 'Analyse')
    
    # Create folders
    if not os.path.exists(path_doseFilm):
        os.makedirs(path_doseFilm)     
    if not os.path.exists(path_analyse):
        os.makedirs(path_analyse)  
        
    # Get files in scan folder
    files = os.listdir(path_scan)
    for file in files:
        if file == 'Thumbs.db':
            continue
        if os.path.isdir(os.path.join(path_scan, file)):
            continue
        img_file = os.path.join(path_scan, file) #AJOUT MG 20190628
        filebase, fileext = os.path.splitext(file)    
        if filebase[-4:-1] == '_00':
#            if filebase[-1] != '1':
#                continue
            filebase = filebase[0:-4]
            
    file_doseFilm = os.path.join(path_doseFilm, filebase + '_doseOpt.tif') 
    report_doseFilm = os.path.join(path_doseFilm, filebase + '_report.pdf')
    ref_dose = path_doseRS
    
    #%% ############################### Perform tiff2dose conversion ########################################
    if tiff_2_dose:
            gaf1 = tiff2dose.Gaf(path=img_file, lut_file=lut_file, info=info, clip=clip, rot90=rot_scan)
#            gaf1 = tiff2dose.Gaf(path=path_scan, lut_file=lut_file, info=info, clip=clip, rot90=rot_scan)
            gaf1.dose_opt.save(file_doseFilm)
            gaf1.publish_pdf(filename=report_doseFilm, open_file=tiff_2_dose_show_pdf)
            
    #%% ############################### Perform analysis ########################################
    if dose_2_analysis:
        ref_dose_factor=1.0
        if type(normalisation) is float:
            film_dose_factor = normalisation
        else:
            film_dose_factor = 1.0
        
        film = analysis.DoseAnalysis(film_dose=file_doseFilm, ref_dose=ref_dose, ref_dose_factor=ref_dose_factor, film_dose_factor=film_dose_factor, flipLR=flipLR, flipUD=flipUD, ref_dose_sum=True, rot90=rot90) 
                
        if normalisation == 'norm_film':
            norm_film_ref_MU = 1000
            norm_film_ref_dose = 973
            norm_film_dose = norm_film_MU / norm_film_ref_MU * norm_film_ref_dose
            film.apply_factor_from_roi(norm_dose=norm_film_dose)
        
        if crop_film:
            film.crop_film()
        
        film.register(shift_x=shift_x, shift_y=shift_y, threshold=10, register_using_gradient=True, markers_center=markers_center)

        if normalisation == 'ref_roi':
            film.apply_factor_from_roi()
    
        #%% Perform analysis 3/3
        doseTA = 3
        distTA = 3
        
        filename= '{}_Facteur{:.2f}_Filtre{}_Gamma{}%-{}mm_report.pdf'.format(filebase, film.film_dose_factor,film_filt, doseTA, distTA)
        fileout=os.path.join(path_analyse, filename)
        print("Analyse en cours...")
        film.analyse(doseTA=doseTA, distTA=distTA, threshold=threshold, norm_val=norm_val, film_filt=film_filt)
        print("")
        print("======================= Gamma 3/3 ============================")
        print("      Gammma {}% {}mm: Taux de passage={:.2f}%; Moyenne={:.2f}".format(doseTA, distTA, film.GammaMap.passRate, film.GammaMap.mean))   
        print("==============================================================")
        film.publish_pdf(fileout, open_file=dose_2_analysis_show_pdf, show_hist=True, show_pass_hist=True, show_varDistTA=False, show_var_DoseTA=False, x=None, y=None)
    
        #%% Perform analysis 2/2
        doseTA = 2
        distTA = 2
        
        filename= '{}_Facteur{:.2f}_Filtre{}_Gamma{}%-{}mm_report.pdf'.format(filebase, film.film_dose_factor,film_filt, doseTA, distTA)
        fileout=os.path.join(path_analyse, filename)
        print("Analyse en cours...")
        film.analyse(doseTA=doseTA, distTA=distTA, threshold=threshold, norm_val=norm_val, film_filt=film_filt)
        print("")
        print("======================= Gamma 2/2 ============================")
        print("      Gammma {}% {}mm: Taux de passage={:.2f}%; Moyenne={:.2f}".format(doseTA, distTA, film.GammaMap.passRate, film.GammaMap.mean))
        print("==============================================================")
        film.show_results()    
        film.publish_pdf(fileout, open_file=dose_2_analysis_show_pdf, show_hist=True, show_pass_hist=True, show_varDistTA=False, show_var_DoseTA=False, x=None, y=None)