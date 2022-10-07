# -*- coding: utf-8 -*-
"""
The dose analysis module performs in-depth comparison from film dose to reference dose image from treatment planning system.

Features:
    - Perform registration by identifying fiducial markers to set isocenter
    - Interactive display of analysis results (gamma map, relative error, dose profiles)
    - Gammap analysis: display gamma map, pass rate, histogram, pass rate vs dose bar graph,
      pass rate vs distance to agreement (fixed dose to agreement),
      pass rate vs dose to agreement (fixed distance to agreement)
    - Publish PDF report
    
Requirements:
    This module is built as an extension to pylinac package.
    Tested with pylinac 2.0.0, which is compatible with python 3.5.
    
Written by Jean-Francois Cabana, copyright 2018
"""

import sys
import numpy as np
import imageRGB
import scipy.ndimage.filters as spf
import copy
import matplotlib.pyplot as plt
import os
from pylinac.core.utilities import is_close
import math
from scipy.signal import medfilt
import pickle

from npgamma import calc_gamma
from math import log10, floor

from pylinac.core import pdf
import io
from reportlab.lib.units import cm
from matplotlib.ticker import FormatStrFormatter
from matplotlib.widgets import RectangleSelector


class DoseAnalysis():
    
    def __init__(self, film_dose=None, ref_dose=None, ref_dose_factor=1, film_dose_factor=1, ref_dose_sum=False, deleteRegion=False):

        if film_dose is not None:
            self.film_dose = imageRGB.load(film_dose)
        if ref_dose is not None:
            self.patientID = imageRGB.getID(ref_dose)
            # If need to add multiple plane dose images, assume all images in folder given by ref_dose
            if ref_dose_sum:
                files = os.listdir(ref_dose)
                img_list = []
                for file in files:
                    img_file = os.path.join(ref_dose, file)
                    filebase, fileext = os.path.splitext(file)
                    if file == 'Thumbs.db':
                        continue
                    if os.path.isdir(img_file):
                        continue
                    img_list.append(imageRGB.load(img_file))
                self.ref_dose = img_list[0]
                new_array = np.stack(tuple(img.array for img in img_list), axis=-1)
                self.ref_dose.array = np.sum(new_array, axis=-1)
            else:
                self.ref_dose = imageRGB.load(ref_dose)

        if ref_dose_factor !=1:
            self.ref_dose.array = ref_dose_factor * self.ref_dose.array

        self.film_dose_factor = film_dose_factor
        if film_dose_factor !=1:
            self.film_dose.array = film_dose_factor * self.film_dose.array
        self.ideal_daily_factor = 1

        # if registration is not None:
        #     self.registration = registration

        self.manual_shift_x = 0 #shift in pixel manually do by the user
        self.manual_shift_y = 0 #shift in pixel manually do by the user
        self.manual_rotation = 0 #rotation in degres manually do by the user

        self.ROI = np.array([[0,0], [len(self.film_dose.array[0]), len(self.film_dose.array)]])
        self.RTD = np.array([[0,0], [0,0]])

        self.deleteRegion = deleteRegion

    def analyse(self, film_filt=0, doseTA=3.0, distTA=3.0, threshold=0.1, norm_val='max', computeIDF = False):
        
        # Save some settings
        self.film_filt = film_filt
        self.doseTA = doseTA
        self.distTA = distTA
        self.threshold = threshold
        self.norm_val = norm_val

        if film_filt:
            self.film_dose.array = medfilt(self.film_dose.array,  kernel_size=(film_filt, film_filt))
        self.GammaMap = self.computeGamma2(doseTA=doseTA, distTA=distTA, threshold=threshold, norm_val=norm_val, computeIDF = computeIDF)
        self.computeDiff()

    def computeDiff(self):
        """ Compute the difference map with the reference image.
            Returns self.DiffMap = film_dose - ref_dose """
        self.DiffMap = imageRGB.ArrayImage(self.film_dose.array - self.ref_dose.array, dpi=self.film_dose.dpi)

        #crop image
        self.DiffMap.array[0:int(self.ROI[0][1])][:] = np.nan
        self.DiffMap.array[int(self.ROI[1][1]):][:] = np.nan
        self.DiffMap.array[:,0:int(self.ROI[0][0])] = np.nan
        self.DiffMap.array[:,int(self.ROI[1][0]):] = np.nan

        self.RelError = imageRGB.ArrayImage(100*self.DiffMap.array/self.ref_dose.array, dpi=self.film_dose.dpi)
        self.DiffMap.MSE =  np.nansum(np.nansum(self.DiffMap.array**2)) / len(self.film_dose.array[(self.film_dose.array > 0)])
        self.DiffMap.RMSE = self.DiffMap.MSE**0.5

    def computeDailyFactor(self):

        #       # error checking
        if not is_close(self.film_dose.dpi, self.ref_dose.dpi, delta=1.5):
            raise AttributeError("The image DPIs to not match: {:.2f} vs. {:.2f}".format(self.film_dose.dpi, self.ref_dose.dpi))
        same_x = is_close(self.film_dose.shape[1], self.ref_dose.shape[1], delta=1.1)
        same_y = is_close(self.film_dose.shape[0], self.ref_dose.shape[0], delta=1.1)
        if not (same_x and same_y):
            raise AttributeError("The images are not the same size: {} vs. {}".format(self.film_dose.shape, self.ref_dose.shape))

        #Erase data outside of ROI
        self.ref_dose.array[0:int(self.ROI[0][1])][:] = np.nan
        self.ref_dose.array[int(self.ROI[1][1]):][:] = np.nan
        self.ref_dose.array[:,0:int(self.ROI[0][0])] = np.nan
        self.ref_dose.array[:,int(self.ROI[1][0]):] = np.nan

        self.film_dose.array[0:int(self.ROI[0][1])][:] = np.nan
        self.film_dose.array[int(self.ROI[1][1]):][:] = np.nan
        self.film_dose.array[:,0:int(self.ROI[0][0])] = np.nan
        self.film_dose.array[:,int(self.ROI[1][0]):] = np.nan

        meanFilm = np.nanmean(self.film_dose.array)
        meanRef = np.nanmean(self.ref_dose.array)
        stdFilm = np.nanstd(self.film_dose.array)
        stdRef = np.nanstd(self.ref_dose.array)
        print("Film mean: ", meanFilm)
        print("Ref mean: ", meanRef)
        print("Film std: ", stdFilm)
        print("Ref std: ", stdRef)
        print("Daily factor: ", 1-(meanFilm-meanRef)/meanRef)

        x = np.floor(self.ref_dose.shape[1] / 2).astype(int)
        y = np.floor(self.ref_dose.shape[0] / 2).astype(int)

        fig, ((ax1, ax2), (ax5, ax6), (ax7, ax8)) = plt.subplots(3, 2, figsize=(10, 8))
        fig.tight_layout()
        fig.canvas.set_window_title('Results')
        axes = [ax1, ax2, ax5, ax6]
        ax7.axis("off")
        ax8.axis("off")

        # Add a table at the bottom of the axes
        data = np.around( [[meanFilm, stdFilm],[meanRef, stdRef]], decimals=4)

        columns = ('Average', 'Standard deviation')
        rows = ('Film', 'Reference')
        DF_header = "Daily Factor = {:.4f}".format(1-(meanFilm-meanRef)/meanRef)

        header = ax7.table(cellText=[['']],
                           colLabels=[DF_header],
                           loc='center',
                           bbox=[0.5, 0.5, 1.0, 0.3])

        the_table = ax7.table(cellText=data,
                              rowLabels=rows,
                              colLabels=columns,
                              loc='center',
                              bbox=[0.5, 0.2, 1.0, 0.45])

        max_dose_comp = np.nanpercentile(self.ref_dose.array, [99.9])[0].round(decimals=3)
        clim = [0, max_dose_comp]

        self.film_dose.plotCB(ax1, clim=clim, title='Film dose (av. = {})'.format(meanFilm), show=False)
        self.ref_dose.plotCB(ax2, clim=clim, title='Reference dose(av. = {})'.format(meanRef), show=False)

        self.show_profiles(axes, x=x, y=y)

        fig.canvas.mpl_connect('button_press_event', lambda event: self.set_profile(event, axes))
        fig.canvas.mpl_connect('motion_notify_event', lambda event: self.moved_and_pressed(event, axes))

        plt.show()

    def computeGamma2(self, doseTA=2, distTA=2, threshold=0.1, norm_val=None, computeIDF = False):
        """Using npgamma
        """
        #       # error checking
        if not is_close(self.film_dose.dpi, self.ref_dose.dpi, delta=1):
            raise AttributeError("The image DPIs to not match: {:.2f} vs. {:.2f}".format(self.film_dose.dpi, self.ref_dose.dpi))
        same_x = is_close(self.film_dose.shape[1], self.ref_dose.shape[1], delta=1.1)
        same_y = is_close(self.film_dose.shape[0], self.ref_dose.shape[0], delta=1.1)
        if not (same_x and same_y):
            raise AttributeError("The images are not the same size: {} vs. {}".format(self.film_dose.shape, self.ref_dose.shape))

        # set up reference and comparison images
        film_dose = imageRGB.ArrayImage(copy.copy(self.film_dose.array))
        ref_dose = imageRGB.ArrayImage(copy.copy(self.ref_dose.array))

        #crop image
        print(self.ROI[0][1])
        film_dose.array[0:int(self.ROI[0][1])][:] = np.nan
        film_dose.array[int(self.ROI[1][1]):][:] = np.nan
        film_dose.array[:,0:int(self.ROI[0][0])] = np.nan
        film_dose.array[:,int(self.ROI[1][0]):] = np.nan

        #compute the ideal daily factor from the average dose of pixels
        # in the ROI with at least the threshold dose

        if computeIDF:

            ref_dose_cropped = imageRGB.ArrayImage(copy.copy(self.ref_dose.array))

            ref_dose_cropped.array[0:int(self.ROI[0][1])][:] = np.nan
            ref_dose_cropped.array[int(self.ROI[1][1]):][:] = np.nan
            ref_dose_cropped.array[:,0:int(self.ROI[0][0])] = np.nan
            ref_dose_cropped.array[:,int(self.ROI[1][0]):] = np.nan
            ref_dose_cropped.array[ref_dose.array < threshold * np.max(ref_dose)] = np.nan

            film_dose_cropped = imageRGB.ArrayImage(copy.copy(film_dose.array))
            film_dose_cropped.array[ref_dose.array < threshold * np.max(ref_dose)] = np.nan

            meanFilm = np.nanmean(film_dose_cropped.array)
            meanRef = np.nanmean(ref_dose_cropped.array)
            self.ideal_daily_factor = (1-(meanFilm-meanRef)/meanRef)*self.film_dose_factor
            print("Film mean: ", meanFilm)
            print("Ref mean: ", meanRef)
            print("Ideal daily factor: ",  self.ideal_daily_factor)

        if norm_val is not None:
            if norm_val is 'max':
                norm_val = ref_dose.array.max()
            film_dose.normalize(norm_val)
            ref_dose.normalize(norm_val)

        # invalidate dose values below threshold so gamma doesn't calculate over it
        ref_dose.array[ref_dose.array < threshold * np.max(ref_dose)] = 0
        film_dose.array[ref_dose.array < threshold * np.max(ref_dose)] = 0

        # convert distance value from mm to pixels
        distTA_pixels = self.film_dose.dpmm * distTA

        # set coordinates (we use pixels since distTA is in pixels)
        x = ref_dose.shape[0]
        y = ref_dose.shape[1]
        x_coord = list(range(0, x))
        y_coord = list(range(0, y))
        coords_reference = (x_coord, y_coord)
        coords_evaluation = (x_coord, y_coord)

        # set thresholds
        distance_threshold = distTA_pixels
        distance_step_size = distance_threshold / 10
        dose_threshold = doseTA / 100 * np.max(ref_dose)
        lower_dose_cutoff = threshold * np.max(ref_dose)

        maximum_test_distance = distance_threshold * 1
        max_concurrent_calc_points = np.inf
        num_threads = 1

        gamma = calc_gamma(coords_reference, ref_dose.array, coords_evaluation, film_dose.array, distance_threshold,
                           dose_threshold, lower_dose_cutoff=lower_dose_cutoff, distance_step_size=distance_step_size,
                           maximum_test_distance=maximum_test_distance,
                           max_concurrent_calc_points=max_concurrent_calc_points, num_threads=num_threads)

        GammaMap = imageRGB.ArrayImage(gamma, dpi=film_dose.dpi)

        fail = np.zeros(GammaMap.shape)
        fail[(GammaMap.array > 1.0)] = 1
        GammaMap.fail = imageRGB.ArrayImage(fail, dpi=film_dose.dpi)

        passed = np.zeros(GammaMap.shape)
        passed[(GammaMap.array <= 1.0)] = 1
        GammaMap.passed = imageRGB.ArrayImage(passed, dpi=film_dose.dpi)

        GammaMap.npassed = sum(sum(passed == 1))
        GammaMap.nfail = sum(sum(fail == 1))
        GammaMap.npixel = GammaMap.npassed + GammaMap.nfail
        GammaMap.passRate = GammaMap.npassed / GammaMap.npixel * 100
        GammaMap.mean = np.nanmean(GammaMap.array)

        return GammaMap

    def plot_gamma_varDoseTA(self, ax=None, start=0.5, stop=4, step=0.5):
        distTA = self.distTA
        threshold = self.threshold
        norm_val = self.norm_val

        values = np.arange(start, stop, step)
        GammaVarDoseTA = np.zeros((len(values), 2))

        i = 0
        for value in values:
            gamma = self.computeGamma2(doseTA=value, distTA=distTA, threshold=threshold, norm_val=norm_val)
            GammaVarDoseTA[i, 0] = value
            GammaVarDoseTA[i, 1] = gamma.passRate
            i = i + 1

        if ax is None:
            fig, ax = plt.subplots()
        x = GammaVarDoseTA[:, 0]
        y = GammaVarDoseTA[:, 1]
        ax.plot(x, y, 'o-')
        ax.set_title('Variable Dose TA, Dist TA = {} mm'.format(distTA))
        ax.set_xlabel('Dose TA (%)')
        ax.set_ylabel('Gamma pass rate (%)')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

        for i in range(0,len(x)):  # <--
            ax.annotate(round(y[i],2), (x[i],y[i]))

    def plot_gamma_varDistTA(self, ax=None, start=0.5, stop=4, step=0.5):
        doseTA = self.doseTA
        threshold = self.threshold
        norm_val = self.norm_val

        values = np.arange(start, stop, step)
        GammaVarDistTA = np.zeros((len(values), 2))

        i = 0
        for value in values:
            gamma = self.computeGamma2(doseTA=doseTA, distTA=value, threshold=threshold, norm_val=norm_val)
            GammaVarDistTA[i, 0] = value
            GammaVarDistTA[i, 1] = gamma.passRate
            i = i + 1

        x = GammaVarDistTA[:, 0]
        y = GammaVarDistTA[:, 1]
        if ax is None:
            fig, ax = plt.subplots()
            fig, ax = plt.subplots()
        ax.plot(x, y, 'o-')
        ax.set_title('Variable Dist TA, Dose TA = {} %'.format(doseTA))
        ax.set_xlabel('Dist TA (mm)')
        ax.set_ylabel('Gamma pass rate (%)')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

        for i in range(0,len(x)):  # <--
            ax.annotate(round(y[i],2), (x[i],y[i]))

    def plot_gamma_hist(self, ax=None, bins='auto', range=[0,3]):
        if ax is None:
            fig, ax = plt.subplots()
        ax.hist(self.GammaMap.array[np.isfinite(self.GammaMap.array)], bins=bins, range=range)
        ax.set_xlabel('Gamma value')
        ax.set_ylabel('Pixels count')
        ax.set_title("Gamma map histogram")
        
    def plot_gamma_pass_hist(self, ax=None, bin_number=20):
        bin_size = round(self.ref_dose.array.max(), -int(floor(log10(abs(self.ref_dose.array.max())))))/bin_number
        if ax is None:
            fig, ax = plt.subplots()
        analyzed = np.isfinite(self.GammaMap.array)
        bins = np.arange(0, self.ref_dose.array.max()+bin_size, bin_size)
        dose = self.ref_dose.array[analyzed]
        gamma_pass = self.GammaMap.passed.array[analyzed]
        dose_pass = (gamma_pass * dose)
        dose_hist = np.histogram(dose, bins=bins)
        dose_pass_hist = np.histogram(dose_pass, bins=bins)
        dose_pass_rel = np.zeros(len(dose_pass_hist[0]))
        
        for i in range(0,len(dose_pass_hist[0])):
            if dose_hist[0][i] > 0:
                dose_pass_rel[i] = float(dose_pass_hist[0][i]) / float(dose_hist[0][i]) * 100
        
        ax.bar(bins[:-1], dose_pass_rel, width=bin_size,  align='edge', linewidth=1, edgecolor='k')
        ax.set_xlabel('Doses (cGy)')
        ax.set_ylabel('Pass rate (%)')
        ax.set_title("Gamma pass rate vs dose")
        for index, bin in enumerate(bins):
            if (index % 2 == 0):
                bins[index] = 0
        ax.set_xticks(bins)

    def plot_gamma_stats(self, figsize=(10, 10), show_hist=True, show_pass_hist=True, show_varDistTA=True,
                         show_var_DoseTA=True):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        axes = (ax1, ax2, ax3, ax4)
        i = 0

        if show_hist:
            self.plot_gamma_hist(ax=axes[i])
            i = i + 1
        if show_pass_hist:
            self.plot_gamma_pass_hist(ax=axes[i])
            i = i + 1
        if show_varDistTA:
            self.plot_gamma_varDistTA(ax=axes[i])
            i = i + 1
        if show_var_DoseTA:
            self.plot_gamma_varDoseTA(ax=axes[i])

    def plot_profile(self, ax=None, profile='x', position=None, title=None):
        film = self.film_dose.array
        ref = self.ref_dose.array
        if profile == 'x':
            if position is None:
                position = np.floor(self.ref_dose.shape[0] / 2).astype(int)
            film_prof = film[position,:]
            ref_prof = ref[position,:]
        elif profile == 'y':
            if position is None:
                position = np.floor(self.ref_dose.shape[1] / 2).astype(int)
            film_prof = film[:,position]
            ref_prof = ref[:,position]
        
        if ax is None:
            fig, ax = plt.subplots()    
        ax.clear()
        ax.plot(film_prof,'r-', linewidth=2)
        ax.plot(ref_prof,'b--', linewidth=2)
        ax.set_title(title)
    
    def show_results(self, fig=None, show=True):

        film_fileName=os.path.basename(self.film_dose.path)
        ref_fileName=os.path.basename(self.ref_dose.path)
        
        x = np.floor(self.ref_dose.shape[1] / 2).astype(int)
        y = np.floor(self.ref_dose.shape[0] / 2).astype(int)
        
        
        fig, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2, figsize=(10, 8))
        fig.tight_layout()
        axes = [ax1,ax2,ax3,ax4,ax5,ax6]
        
        max_dose_film = np.percentile(self.film_dose.array[10:40],[99.9])[0].round(decimals=-1)
        max_dose_comp = np.percentile(self.ref_dose.array,[99.9])[0].round(decimals=3)
        clim = [0,  max_dose_comp]

        self.film_dose.plotCB(ax1, clim=clim, title='Film dose ({})'.format(film_fileName), show=False)
        self.ref_dose.plotCB(ax2, clim=clim, title='Reference dose ({})'.format(ref_fileName), show=False)
        self.GammaMap.plotCB(ax3, clim=[0,2], cmap='bwr', title='Gamma map ({:.2f}% pass; {:.2f} mean)'.format(self.GammaMap.passRate, self.GammaMap.mean), show=False)
        
        min_value = max(-10, np.nanpercentile(self.RelError.array,[1])[0].round(decimals=0))
        max_value = min(10, np.nanpercentile(self.RelError.array,[99])[0].round(decimals=0))
        clim = [min_value, max_value]
        self.RelError.plotCB(ax4, cmap='jet', clim=clim, title='Relative Error (%) (RMSE={:.2f})'.format(self.DiffMap.RMSE), show=False)
        self.show_profiles(axes, x=x, y=y)
        
        fig.canvas.mpl_connect('button_press_event', lambda event: self.set_profile(event, axes))
        fig.canvas.mpl_connect('motion_notify_event', lambda event: self.moved_and_pressed(event, axes))
        if show:
            plt.show()

    def show_profiles(self, axes, x, y):
        self.plot_profile(ax=axes[-2], profile='x', title='Horizontal profile (y={})'.format(y), position=y)
        self.plot_profile(ax=axes[-1], profile='y', title='Vertical profile (x={})'.format(x), position=x)
        
        for i in range(0,len(axes)-2):
            ax = axes[i]
            while len(ax.lines) > 0:
                ax.lines[-1].remove()
            
            ax.plot((x,x),(0,self.ref_dose.shape[0]),'w--', linewidth=1)
            ax.plot((0,self.ref_dose.shape[1]),(y,y),'w--', linewidth=1)
        
    def set_profile(self, event, axes):
        x = int(event.xdata)
        y = int(event.ydata)
        self.show_profiles(axes,x=x, y=y)
        plt.gcf().canvas.draw_idle()
        
    def moved_and_pressed(self, event, axes):
        if event.button==1:
            self.set_profile(event, axes)  
###
    def register(self, shift_x=0, shift_y=0, flipLR=False, flipUD=False, rot90=True, register_using_gradient = True, threshold=10, markers_center=None):
        self.register_using_gradient = register_using_gradient
        if flipLR:
            self.film_dose.array = np.fliplr(self.film_dose.array)
        if flipUD:
            self.film_dose.array = np.flipud(self.film_dose.array)
        if rot90:
            self.film_dose.array = np.rot90(self.film_dose.array)

        self.shifts = [shift_x, shift_y]
        self.markers_center = markers_center
        self.film_dose.crop_edges(threshold=threshold)
        self.film_dose.plot(show=False)
        self.select_markers()
        print('Please double-click on each marker, starting from top and going clockwise. Press ''enter'' when done')
        plt.imshow(self.film_dose.array, cmap='inferno')
        plt.clim(0, 600);
        plt.show()

    def select_markers(self):
        fig = plt.gcf()
        fig.canvas.set_window_title('Markers Selection')
        self.markers = []
        ax = plt.gca()
        ax.set_title('Please double-click on each marker, starting from top and going clockwise. \n Press enter when done. \n Top=  ; Right=  ; Bottom=  ; Left=  ')
        fig.canvas.mpl_connect('button_press_event', self.onclick)
        fig.canvas.mpl_connect('key_press_event', self.ontype)

    def onclick(self, event):
        ax = plt.gca()
        if event.dblclick:
            print('Double click!')
            self.markers.append([int(event.xdata), int(event.ydata)])
            if len(self.markers)==1:
                print(1)
                ax.set_title('Top= {}; Right=  ; Bottom=  ; Left=  '.format(self.markers[0]))
            if len(self.markers)==2:
                ax.set_title('Top= {}; Right= {}; Bottom=  ; Left=  '.format(self.markers[0], self.markers[1]))
            if len(self.markers)==3:
                ax.set_title('Top= {}; Right= {}; Bottom= {}; Left=  '.format(self.markers[0], self.markers[1], self.markers[2]))
            if len(self.markers)==4:
                ax.set_title('Top= {}; Right= {}; Bottom= {}; Left= {}'.format(self.markers[0], self.markers[1], self.markers[2], self.markers[3]))

    def ontype(self, event):
        if event.key == 'enter':
            if len(self.markers) != 4:
                print('')
                print('Please start over...')
                print('{} markers were selected when 4 were expected...'.format(len(self.markers)))
                print('Please double-click on each marker, starting from top and going clockwise. Press ''enter'' when done')
                self.markers = []
                ax = plt.gca()
                ax.set_title('Top=  ; Right=  ; Bottom=  ; Left=  ')
            else:
                plt.close(plt.gcf())
                self.move_iso_center()
                self.remove_rotation()
                self.apply_shifts_ref()
                film_dose_path = self.film_dose.path
                ref_dose_path = self.ref_dose.path
                (self.film_dose, self.ref_dose) = imageRGB.equate_images(self.film_dose, self.ref_dose)
                self.film_dose.path = film_dose_path
                self.ref_dose.path = ref_dose_path
                self.tune_registration()
                self.selectROI()
                if(self.deleteRegion):
                    self.selectRegionToDelete()

    def move_iso_center(self):
        x = [ self.markers[0][0], self.markers[1][0], self.markers[2][0], self.markers[3][0] ]
        y = [ self.markers[0][1], self.markers[1][1], self.markers[2][1], self.markers[3][1] ]

        # Find intersection -> isocenter
        line1 = ((x[0],y[0]),(x[2],y[2]))
        line2 = ((x[1],y[1]),(x[3],y[3]))
        (x0,y0) = line_intersection(line1, line2)

        # Find intersection -> (x0, y0)
        line1 = ((x[0], y[0]), (x[2], y[2]))
        line2 = ((x[1], y[1]), (x[3], y[3]))
        (x0, y0) = line_intersection(line1, line2)

        self.x0 = int(np.around(x0))
        self.y0 = int(np.around(y0))

        # Make the (x0, y0) the center of image by padding
        self.film_dose.move_pixel_to_center(x0, y0)

        # Find the corresponding position in the reference image and make it the center
        if self.markers_center is not None:
            self.ref_dose.position = [float(i) for i in self.ref_dose.metadata.ImagePositionPatient]
            self.ref_dose.sizeX = self.ref_dose.metadata.Columns
            self.ref_dose.sizeY = self.ref_dose.metadata.Rows
            self.ref_dose.orientation = self.ref_dose.metadata.SeriesDescription

            if 'Transversal' in self.ref_dose.orientation:
                x_corner = self.ref_dose.position[0]
                y_corner = -1.0 * self.ref_dose.position[1]
                x_marker = self.markers_center[0]
                y_marker = self.markers_center[2]
                x_pos_mm = x_marker - x_corner
                y_pos_mm = y_corner - y_marker
                x0 = int(np.around(x_pos_mm * self.ref_dose.dpmm))
                y0 = int(np.around(y_pos_mm * self.ref_dose.dpmm))

            if 'Sagittal' in self.ref_dose.orientation:
                x_corner = -1.0 * self.ref_dose.position[1]
                y_corner = self.ref_dose.position[2]
                x_marker = self.markers_center[2]
                y_marker = self.markers_center[1]
                x_pos_mm = x_marker - x_corner
                y_pos_mm = y_marker - y_corner
                x0 = self.ref_dose.sizeX + int(np.around(x_pos_mm * self.ref_dose.dpmm))
                y0 = self.ref_dose.sizeY - int(np.around(y_pos_mm * self.ref_dose.dpmm))

            if 'Coronal' in self.ref_dose.orientation:
                x_corner = self.ref_dose.position[0]
                y_corner = self.ref_dose.position[2]
                x_marker = self.markers_center[0]
                y_marker = self.markers_center[1]
                x_pos_mm = x_marker - x_corner
                y_pos_mm = y_marker - y_corner
                x0 = int(np.around(x_pos_mm * self.ref_dose.dpmm))
                y0 = self.ref_dose.sizeY - int(np.around(y_pos_mm * self.ref_dose.dpmm))

            self.ref_dose.move_pixel_to_center(x0, y0)

    def remove_rotation(self):
        x = [ self.markers[0][0], self.markers[1][0], self.markers[2][0], self.markers[3][0] ]
        y = [ self.markers[0][1], self.markers[1][1], self.markers[2][1], self.markers[3][1] ]

        # Find rotation angle
        angle1 = math.degrees( math.atan( (x[2]-x[0]) / (y[2]-y[0]) ) )
        angle2 = math.degrees( math.atan( (y[3]-y[1]) / (x[1]-x[3]) ) )

        # Appy inverse rotation
        angleCorr = -1.0*(angle1+angle2)/2
        print('Applying a rotation of {} degrees'.format(angleCorr))
        self.film_dose.rotate(angleCorr)

    def apply_shifts(self):
        shift_x_pixels = int(round(self.shifts[0] * self.film_dose.dpmm ))
        shift_y_pixels = int(round(self.shifts[1] * self.film_dose.dpmm ))

        if shift_x_pixels > 0:
            self.film_dose.pad(pixels=shift_x_pixels, value=0, edges='left')
        if shift_x_pixels < 0:
            self.film_dose.pad(pixels=abs(shift_x_pixels), value=0, edges='right')
        if shift_y_pixels > 0:
            self.film_dose.pad(pixels=shift_y_pixels, value=0, edges='top')
        if shift_y_pixels < 0:
            self.film_dose.pad(pixels=abs(shift_y_pixels), value=0, edges='bottom')

    def apply_shifts_ref(self):
        print('apply_shifts_ref')
        # Make the isocenter position the center of ref image
        pad_x_pixels =  int(round(self.shifts[0] * self.ref_dose.dpmm *2))
        pad_y_pixels =  int(round(self.shifts[1] * self.ref_dose.dpmm *2))
        print("Applying shifts of {} pixels in x and {} pixels in y.".format(pad_x_pixels, pad_y_pixels))

        if pad_x_pixels > 0:
            self.ref_dose.pad(pixels=pad_x_pixels, value=0, edges='left')
        if pad_x_pixels < 0:
            self.ref_dose.pad(pixels=abs(pad_x_pixels), value=0, edges='right')
        if pad_y_pixels > 0:
            self.ref_dose.pad(pixels=pad_y_pixels, value=0, edges='top')
        if pad_y_pixels < 0:
            self.ref_dose.pad(pixels=abs(pad_y_pixels), value=0, edges='bottom')

    def tune_registration(self):
        sys.setrecursionlimit(2000)
        print('Fine tune registration using keyboard if needed. Arrow keys = move; ctrl+left/right = rotate. Press enter when done.')
        fig = plt.figure()
        ax = plt.gca()

        fig.canvas.mpl_connect('key_press_event', self.reg_ontype)
        self.show_registration(ax=ax)

    def show_registration(self, ax=None):
        fig = plt.gcf()
        fig.canvas.set_window_title('Fine Tune Registration')
        ax.clear()

        if self.register_using_gradient:
            ref_x = spf.sobel(self.ref_dose.as_type(np.float32), 1)
            ref_y = spf.sobel(self.ref_dose.as_type(np.float32), 0)
            ref_grad = np.hypot(ref_x, ref_y)
            film_x = spf.sobel(self.film_dose.as_type(np.float32), 1)
            film_y = spf.sobel(self.film_dose.as_type(np.float32), 0)
            film_grad = np.hypot(film_x, film_y)
            img_array = film_grad - ref_grad

        img_array = self.film_dose.array - self.ref_dose.array
        img = imageRGB.load(img_array, dpi=self.film_dose.dpi)
        img.ground()
#        img.normalize()
#        img = self.film_dose
        img.plot(ax=ax, show=False)
        ax.plot((0, img.shape[1]), (img.center.y, img.center.y),'k--')
        ax.plot((img.center.x, img.center.x), (0, img.shape[0]),'k--')
        ax.set_xlim(0, img.shape[1])
        ax.set_ylim(img.shape[0],0)
        ax.set_title('Fine tune registration. \n Arrow keys = move; ctrl+left/right = rotate. Press enter when done.')
        plt.imshow(img_array, cmap='inferno')
        plt.clim(np.percentile(img_array,[0.1]), np.percentile(img_array,[95]));
        plt.show()

    def reg_ontype(self, event):
        fig = plt.gcf()
        ax = plt.gca()
        #print(event.key)
        if event.key == 'up':
            self.manual_shift_y -=1
            self.film_dose.roll(direction='y', amount=-1)
            self.show_registration(ax=ax)
            fig.canvas.draw_idle()
        if event.key == 'down':
            self.manual_shift_y += +1
            self.film_dose.roll(direction='y', amount=1)
            self.show_registration(ax=ax)
            fig.canvas.draw_idle()
        if event.key == 'left':
            self.manual_shift_x -=1
            self.film_dose.roll(direction='x', amount=-1)
            self.show_registration(ax=ax)
            fig.canvas.draw_idle()
        if event.key == 'right':
            self.manual_shift_x +=1
            self.film_dose.roll(direction='x', amount=1)
            self.show_registration(ax=ax)
            fig.canvas.draw_idle()
        if event.key == 'ctrl+right':
            self.manual_rotation +=-0.1
            self.film_dose.rotate(-0.1)
            self.show_registration(ax=ax)
            fig.canvas.draw_idle()
        if event.key == 'ctrl+left':
            self.manual_rotation +=0.1
            self.film_dose.rotate(0.1)
            self.show_registration(ax=ax)
            fig.canvas.draw_idle()
        if event.key == 'enter':
            print("X shift:", self.manual_shift_x)
            print("Y shift:", self.manual_shift_y)
            print("Rotation:", self.manual_rotation)
            plt.close(plt.gcf())

    #Functions for the ROI selector

    def selectROI(self):

        def line_select_callback(eclick, erelease):
            'eclick and erelease are the press and release events'
            self.ROI[0][0], self.ROI[0][1] = int(eclick.xdata), int(eclick.ydata)
            self.ROI[1][0], self.ROI[1][1] = int(erelease.xdata), int(erelease.ydata)

        self.film_dose.plot(show=False)
        print('dpmm: ', self.film_dose.dpmm)
        print('Select the region of interest where to perform the analysis. Press enter when done.')
        fig = plt.gcf()
        ax = plt.gca()

        ax.set_title('Select the region of interest where to perform the analysis. \n Press enter when done. \n If no ROI is selected, the whole image will be analyzed')

        self.rs = RectangleSelector(ax, line_select_callback,
                                               drawtype='box', useblit=True,
                                               button=[1, 3],  # don't use middle button
                                               minspanx=5, minspany=5,
                                               spancoords='pixels',
                                               interactive=True)

        fig.canvas.mpl_connect('key_press_event', self.press_enter)
        plt.imshow(self.film_dose.array, cmap='inferno')
        plt.clim(np.percentile(self.film_dose.array, [0.1]), np.percentile(self.film_dose.array, [95]));
        fig.canvas.set_window_title('ROI Selection')
        plt.show()

    def selectRegionToDelete(self):

        def line_select_callback(eclick, erelease):
            'eclick and erelease are the press and release events'
            self.RTD[0][0], self.RTD[0][1] = int(eclick.xdata), int(eclick.ydata)
            self.RTD[1][0], self.RTD[1][1] = int(erelease.xdata), int(erelease.ydata)

        self.film_dose.plot(show=False)
        print('Select the region to delete. Press enter when done.')
        fig = plt.gcf()
        ax = plt.gca()

        ax.set_title('Select the region to delete. \n Press the Press enter when done. ')

        self.rs = RectangleSelector(ax, line_select_callback,
                                               drawtype='box', useblit=True,
                                               button=[1, 3],  # don't use middle button
                                               minspanx=5, minspany=5,
                                               spancoords='pixels',
                                               interactive=True)

        fig.canvas.mpl_connect('key_press_event', self.press_enter_RTD)
        plt.imshow(self.film_dose.array, cmap='inferno')
        plt.clim(np.percentile(self.film_dose.array, [0.1]), np.percentile(self.film_dose.array, [95]));
        fig.canvas.set_window_title('Delete Region')
        plt.show()

    def press_enter(self, event):
        """ Continue LUT creation when ''enter'' is pressed. """

        if event.key == 'enter':
            plt.close(plt.gcf())
            del self.rs

    def press_enter_RTD(self, event):
        if event.key == 'enter':
            plt.close(plt.gcf())
            del self.rs
        print(self.RTD[0][0])
        print(np.nanmean(self.film_dose.array))
        #self.film_dose.array[int(self.RTD[0][0]):int(self.RTD[0][1])][int(self.RTD[1][0]):int(self.RTD[1][1])] = np.nan
        #self.film_dose.array[int(self.RTD[0][1]):int(self.RTD[1][1])][:] = np.nan
        self.film_dose.array[int(self.RTD[0][1]):int(self.RTD[1][1]),int(self.RTD[0][0]):int(self.RTD[1][0])] = np.nan
        print(np.nanmean(self.film_dose.array))


    ###QUASAR registration

    def registerQUASAR(self, shift_x=0, shift_y=0, flipLR=False, flipUD=False, threshold=10, displacement_longi=0):
        #Function to register when using the QUASAR phantom

        self.displacement_longi = displacement_longi
        if flipLR:
            self.film_dose.array = np.fliplr(self.film_dose.array)
        if flipUD:
            self.film_dose.array = np.flipud(self.film_dose.array)
        self.shifts = [shift_x, shift_y]
        self.film_dose.crop_edges(threshold=threshold)
        self.film_dose.plot(show=False)
        self.select_markersQUASAR()
        print('Please double-click on the two markers of the QUASAR, from left to right. Press ''enter'' when done')
        plt.show()

    def select_markersQUASAR(self):
        fig = plt.gcf()
        fig.canvas.set_window_title('Markers Selection')
        self.markers = []
        ax = plt.gca()
        ax.set_title('Top=  ; Right=  ; Bottom=  ; Left=  ')
        fig.canvas.mpl_connect('button_press_event', self.onclick)
        fig.canvas.mpl_connect('key_press_event', self.ontypeQUASAR)


    def onclickQUASAR(self, event):
        ax = plt.gca()
        if event.dblclick:
            self.markers.append([int(event.xdata), int(event.ydata)])
            if len(self.markers)==1:
                ax.set_title('Top= {}; Right=  ; Bottom=  ; Left=  '.format(self.markers[0]))
            if len(self.markers)==2:
                ax.set_title('Top= {}; Right= {}; Bottom=  ; Left=  '.format(self.markers[0], self.markers[1]))

    def ontypeQUASAR(self, event):
        if event.key == 'enter':

            if len(self.markers) != 2:
                print('')
                print('Please start over...')
                print('{} markers were selected when 2 were expected...'.format(len(self.markers)))
                print('Please double-click on each marker. Press ''enter'' when done')
                self.markers = []
                ax = plt.gca()
                ax.set_title('Top=  ; Right=  ; Bottom=  ; Left=  ')
            else:
                plt.close(plt.gcf())
                self.move_iso_centerQUASAR()
                self.apply_shifts()
                film_dose_path = self.film_dose.path
                ref_dose_path = self.ref_dose.path
                (self.film_dose, self.ref_dose) = imageRGB.equate_images(self.film_dose, self.ref_dose)
                self.film_dose.path = film_dose_path
                self.ref_dose.path = ref_dose_path
                self.tune_registration()
                self.selectROI()

    def move_iso_centerQUASAR(self):
        x = [self.markers[0][0], self.markers[1][0]]
        y = [self.markers[0][1], self.markers[1][1]]

        dist_pin2iso = (70 - self.displacement_longi)*self.film_dose.dpmm

        # Find the isocenter and the rotation to apply
        line_center = ((x[0]+x[1])/2,(y[0]+y[1])/2)
        print("markers: ", self.markers)
        print("line_center: ",line_center)
        theta = math.atan((y[0]-y[1])/(x[1]-x[0]))
        print("theta: ", theta*180/math.pi)

        isocenter  = (line_center[0] + math.sin(theta)*dist_pin2iso, line_center[1] + math.cos(theta)*dist_pin2iso)
        print("math.cos(theta)*dist_pin2iso: ", math.cos(theta)*dist_pin2iso)
        print("isocenter:", isocenter)

        # Make the isocenter the center of image by padding
        left = isocenter[0]
        right = self.film_dose.shape[1] - isocenter[0]
        top = isocenter[1]
        bottom = self.film_dose.shape[0] - isocenter[1]
        if left < right:
            self.film_dose.pad(pixels = int(right-left), edges='left')
        else:
            self.film_dose.pad(pixels = int(left-right), edges='right')
        if top < bottom:
            self.film_dose.pad(pixels = int(bottom-top), edges='top')
        else:
            self.film_dose.pad(pixels = int(top-bottom), edges='bottom')

        self.remove_rotationQUASAR(angle=theta)

    def remove_rotationQUASAR(self, angle=0):

        print('Applying a rotation of {} degrees'.format(math.degrees(angle)))
        self.film_dose.rotate(-math.degrees(angle))

    # def apply_shifts(self):
    #     shift_x_pixels =  int(round(self.shifts[0] * self.film_dose.dpmm ))
    #     shift_y_pixels =  int(round(self.shifts[1] * self.film_dose.dpmm ))
    #
    #     if shift_x_pixels > 0:
    #         self.film_dose.pad(pixels=shift_x_pixels, value=0, edges='left')
    #     if shift_x_pixels < 0:
    #         self.film_dose.pad(pixels=abs(shift_x_pixels), value=0, edges='right')
    #     if shift_y_pixels > 0:
    #         self.film_dose.pad(pixels=shift_y_pixels, value=0, edges='top')
    #     if shift_y_pixels < 0:
    #         self.film_dose.pad(pixels=abs(shift_y_pixels), value=0, edges='bottom')

    ###############

    def save_analyzed_image(self, filename, **kwargs):
        """Save the analyzed image to a file.

        Parameters
        ----------
        filename : str
            The location and filename to save to.
        kwargs
            Keyword arguments are passed to plt.savefig().
        """
        self.show_results(show=False)
        fig = plt.gcf()
        fig.savefig(filename)
        plt.close(fig)
        
    def save_analyzed_gamma(self, filename, show_hist=True, show_pass_hist=True, show_varDistTA=False, show_var_DoseTA=False, **kwargs):
        """Save the analyzed image to a file.

        Parameters
        ----------
        filename : str
            The location and filename to save to.
        kwargs
            Keyword arguments are passed to plt.savefig().
        """
        self.plot_gamma_stats(show_hist=show_hist, show_pass_hist=show_pass_hist, show_varDistTA=show_varDistTA, show_var_DoseTA=show_var_DoseTA)
        fig = plt.gcf()
        fig.savefig(filename)
        plt.close(fig)
            
    def publish_pdf(self, filename=None, author=None, unit=None, notes=None, open_file=False, show_hist=True, show_pass_hist=True, show_varDistTA=False, show_var_DoseTA=False, **kwargs):
        """Publish a PDF report of the calibration. The report includes basic
        file information, the image and determined ROIs, and the calibration curves

        Parameters
        ----------
        filename : str
            The path and/or filename to save the PDF report as; must end in ".pdf".
        author : str, optional
            The person who analyzed the image.
        unit : str, optional
            The machine unit name or other identifier (e.g. serial number).
        notes : str, list of strings, optional
            If a string, adds it as a line of text in the PDf report.
            If a list of strings, each string item is printed on its own line. Useful for writing multiple sentences.
        """
        if filename is None:
            filename = os.path.join(self.path, 'Report.pdf')
        title='Film Analysis Report'
        canvas = pdf.create_pylinac_page_template(filename, analysis_title=title)
        
        data = io.BytesIO()
        self.save_analyzed_image(data)
        img = pdf.create_stream_image(data)
        canvas.drawImage(img, 0.5 * cm, 2 * cm, width=20 * cm, height=20 * cm, preserveAspectRatio=True)
        
        pdf.draw_text(canvas, x=1 * cm, y=25.5 * cm, text='Analysis infos:', fontsize=12)
        text = ['Film dose: {}'.format(self.film_dose.path),
                'Reference dose: {}'.format(self.ref_dose.path),
                'Film filter kernel: {}'.format(self.film_filt),
                'Gamma threshold: {}'.format(self.threshold),
                'Gamma dose-to-agreement: {}'.format(self.doseTA),
                'Gamma distance-to-agreement: {}'.format(self.distTA),
                'Gamma normalization: {}'.format(self.norm_val),
                'Manual displacements: x={} mm, y={} mm, theta={}°'. format(self.manual_shift_x/self.film_dose.dpmm, self.manual_shift_y/self.film_dose.dpmm, self.manual_rotation),
                'Film dose factor: {}'. format(self.film_dose_factor),
                'Ideal film dose factor: {}'. format(self.ideal_daily_factor),
                'Patient ID: {}'.format(self.patientID)
                ]
        pdf.draw_text(canvas, x=1 * cm, y=25 * cm, text=text, fontsize=10)

        if(show_hist or show_pass_hist or show_varDistTA or show_var_DoseTA):
            canvas.showPage()

            pdf.add_pylinac_page_template(canvas, analysis_title=title)
            pdf.draw_text(canvas, x=1 * cm, y=25.5 * cm, text='Analysis infos:', fontsize=12)
            pdf.draw_text(canvas, x=1 * cm, y=25 * cm, text=text, fontsize=10)
            data = io.BytesIO()
            self.save_analyzed_gamma(data, figsize=(10, 10), show_hist=show_hist, show_pass_hist=show_pass_hist, show_varDistTA=show_varDistTA, show_var_DoseTA=show_var_DoseTA, **kwargs)
            img = pdf.create_stream_image(data)
            canvas.drawImage(img, 0.5*cm, 2*cm, width=20*cm, height=20*cm, preserveAspectRatio=True)
#        canvas.showPage()
        
        if notes is not None:
            pdf.draw_text(canvas, x=1 * cm, y=2.5 * cm, fontsize=14, text="Notes:")
            pdf.draw_text(canvas, x=1 * cm, y=2 * cm, text=notes)
        pdf.finish(canvas, open_file=open_file, filename=filename)


########################### End class DoseAnalysis ##############################
    
def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]) #Typo was here

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def save_dose(dose, filename):
    dose.filename = filename
    with open(filename, 'wb') as output:
        pickle.dump(dose, output, pickle.HIGHEST_PROTOCOL)

def load_dose(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)

