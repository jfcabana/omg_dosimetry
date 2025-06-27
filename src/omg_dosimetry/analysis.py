# -*- coding: utf-8 -*-
"""
OMG Dosimetry analysis module.

The dose analysis module performs in-depth comparison from film dose to reference dose image from treatment planning system.

Features:
    - Perform registration by identifying fiducial markers on the film,
    - Interactive display of analysis results (gamma map, relative error, dose profiles)
    - Gamma analysis: display gamma map, pass rate, histogram, pass rate vs dose bar graph,
      pass rate vs distance to agreement (fixed dose to agreement),
      pass rate vs dose to agreement (fixed distance to agreement)
    - Publish PDF report
    
Written by Jean-Francois Cabana, copyright 2018
Modified by Peter Truong (CISSSO)
Version: 2025-06-27
"""

import numpy as np
import scipy.ndimage.filters as spf
import copy
import matplotlib.pyplot as plt
import os
from pylinac.core.utilities import is_close
import math
from scipy.signal import medfilt
import pickle
from pylinac.core import pdf
import io
from pathlib import Path
import pymedphys
from matplotlib.widgets  import RectangleSelector, MultiCursor, Cursor
import webbrowser
from .imageRGB import load, ArrayImage, equate_images
import bz2
import time
from .tools import Ruler

class DoseAnalysis(): 
    """
    Base class for analysis film dose vs reference dose.

    Usage:
    -------
    film = analysis.DoseAnalysis(film_dose=file_doseFilm, ref_dose=ref_dose)

    Attributes
    ----------
    film_dose : str
        File path of planar dose image of the scanned film converted to dose (using tiff2dose module).

    ref_dose : str
        File path of the reference dose (from TPS).
        
    norm_film_dose : str, optional, default=None
        File path of the normalization film dose if scanned separately. Principle being that the same 
        normalization film scan can be used for other tif images of film scanned at the same time.

    film_dose_factor : float, optional, default=1.0
        Scaling factor to apply to the film dose.

    ref_dose_factor : float, optional, default=1.0
        Scaling factor to apply to the reference dose.

    flipLR : bool, optional, default=False
        Whether or not to flip the film dose horizontally to match reference dose orientation.

    flipUD : bool, optional, default=False
        Whether or not to flip the film dose vertically to match reference dose orientation.

    rot90 : int, optional, default=0
        If not 0, number of 90 degrees rotation to apply to the film to match reference dose orientation.

    ref_dose_sum : bool, optional, default=False
        If True, all planar dose files found in the ref_dose folder will be summed together.
        
    apply_dose_factors : bool, optional, default = True
        If True, apply parameters film_dose_factor and ref_dose_factor to film/reference dose maps.
    """

    def __init__(self, film_dose=None, ref_dose=None, norm_film_dose = None, film_dose_factor=1, ref_dose_factor=1, flipLR=False, flipUD=False, rot90=0, ref_dose_sum=False, apply_dose_factors = True):
        self.film_dose = load(film_dose) if film_dose else None
        self.norm_film_dose = load(norm_film_dose) if norm_film_dose else None        
        self.ref_dose = self.load_reference_dose(ref_dose, ref_dose_sum) if ref_dose else None
        if apply_dose_factor: 
            self.apply_film_factor(film_dose_factor)
            self.apply_ref_factor(ref_dose_factor)
        else:       # Retain initialization property for DoseAnalysis object
            self.film_dose_factor = film_dose_factor
            self.ref_dose_factor = ref_dose_factor
        if self.film_dose:
            if rot90: self.film_dose.array = np.rot90(self.film_dose.array, k=rot90)
            if flipLR: self.film_dose.array = np.fliplr(self.film_dose.array)
            if flipUD: self.film_dose.array = np.flipud(self.film_dose.array)

    def load_reference_dose(self, ref_dose_path, ref_dose_sum):
        if ref_dose_sum:
            # If needed to sum multiple plane dose images, assume all images in folder given by ref_dose_path
            img_list = [load(os.path.join(ref_dose_path, file)) for file in os.listdir(ref_dose_path) if file != 'Thumbs.db' and not os.path.isdir(os.path.join(ref_dose_path, file))]
            combined_array = np.stack([img.array for img in img_list], axis=-1)
            ref_dose = img_list[0]
            ref_dose.array = np.sum(combined_array, axis=-1)
            return ref_dose
        return load(ref_dose_path)

    def apply_film_factor(self, film_dose_factor = None):
        """ Apply a normalisation factor to film dose. """
        if film_dose_factor:
            self.film_dose_factor = film_dose_factor
            self.film_dose.array *= film_dose_factor
            print(f"\nApplied film normalisation factor = {film_dose_factor:.2f}")

    def apply_ref_factor(self, ref_dose_factor = None):
        """ Apply a normalisation factor to reference dose. """
        if ref_dose_factor is not None:
            self.ref_dose_factor = ref_dose_factor
            self.ref_dose.array *= ref_dose_factor
            print(f"Applied ref dose normalisation factor = {ref_dose_factor:.2f}")

    def apply_factor_from_isodose(self, norm_isodose = 0):
        """ Apply film normalisation factor from a reference dose isodose [cGy].
            Mean dose inside regions where ref_dose > norm_isodose will be compared
            between film and ref_dose. A factor is computed and applied to film dose
            so that median dose in this region is the same for both.
        """
        print(f"Computing normalisation factor from doses > {norm_isodose} cGy.")
        self.norm_dose = norm_isodose        
        indices = np.where(self.ref_dose.array > self.norm_dose)   
        self.apply_film_factor(np.median(self.ref_dose.array[indices]) / np.median(self.film_dose.array[indices]))
        
    def apply_factor_from_roi(self, norm_dose=None, apply=True):
        """ Apply film normalisation factor from a rectangle ROI.
            Brings up an interactive plot, where the user must define a rectangle ROI
            that will be used to compute a film normalisation factor.
            Median dose inside this rectangle will be used to scale the film dose to match
            that of the reference.
        """
        self.norm_dose = norm_dose      
        msg = '\nFactor from ROI: Click and drag to draw an ROI manually. Press ''enter'' when finished.'
        self.roi_xmin, self.roi_xmax, self.roi_ymin, self.roi_ymax = [], [], [], []

        self.fig = plt.figure()
        ax = plt.gca()  
        (self.norm_film_dose if self.norm_film_dose else self.film_dose).plot(ax=ax)
        ax.set_title(msg)
        print(msg)
        
        def select_box(eclick, erelease):
            x1, y1 = int(eclick.xdata), int(eclick.ydata)
            x2, y2 = int(erelease.xdata), int(erelease.ydata)
            self.roi_xmin, self.roi_xmax = min(x1,x2), max(x1,x2)
            self.roi_ymin, self.roi_ymax = min(y1,y2), max(y1,y2)
        
        self.rs = RectangleSelector(ax, select_box, useblit=True, button=[1], minspanx=5, minspany=5, spancoords='pixels', interactive=True)  
        if apply: self.cid = self.fig.canvas.mpl_connect('key_press_event', self.apply_factor_from_roi_press_enter)
        else: self.cid = self.fig.canvas.mpl_connect('key_press_event', self.get_factor_from_roi_press_enter)
        
        self.wait = True
        while self.wait: plt.pause(1)
        self.cleanup()

    def get_factor_from_roi_press_enter(self, event):
        """ Function called from apply_factor_from_roi() when ''enter'' is pressed. """      
        if event.key == 'enter':
            roi_film = np.median(self.film_dose.array[self.roi_ymin:self.roi_ymax, self.roi_xmin:self.roi_xmax])
            roi_ref = np.median(self.ref_dose.array[self.roi_ymin:self.roi_ymax, self.roi_xmin:self.roi_xmax])
            relative_diff = (roi_film - roi_ref) / roi_ref * 100
            print(f"Median film dose = {roi_film:.1f} cGy; median ref dose = {roi_ref:.1f} cGy; Relative diff = {relative_diff:.1f}%")
            self.wait = False

    def apply_factor_from_roi_press_enter(self, event):
        """ Function called from apply_factor_from_roi() when ''enter'' is pressed. """      
        if event.key == 'enter':
            film_dose_array = self.norm_film_dose.array if self.norm_film_dose else self.film_dose.array
            roi_film = np.median(film_dose_array[self.roi_ymin:self.roi_ymax, self.roi_xmin:self.roi_xmax])
            
            if self.norm_dose is None:  # If no normalisation dose is given, assume normalisation is on ref_dose
                roi_ref = np.median(self.ref_dose.array[self.roi_ymin:self.roi_ymax, self.roi_xmin:self.roi_xmax])
                factor = roi_ref / roi_film
                print(f"Median film dose = {roi_film:.1f} cGy; median ref dose = {roi_ref:.1f} cGy")    
            else: factor = self.norm_dose / roi_film            
            self.apply_film_factor(film_dose_factor = factor)
            self.wait = False

    def apply_factor_from_norm_film(self, norm_dose = None, norm_roi_size = 10):
        """ Define an ROI of norm_roi_size mm x norm_roi_size mm to compute dose factor from a normalisation film (in the same scan). """
        
        self.norm_dose = norm_dose
        self.norm_roi_size = norm_roi_size
        msg = '\nFactor from normalisation film: Double-click at the center of the film markers. Press enter when done'
        self.roi_center, self.roi_xmin, self.roi_xmax, self.roi_ymin, self.roi_ymax = [], [], [], [], []
        
        self.fig = plt.figure()
        ax = plt.gca()  
        self.film_dose.plot(ax=ax)  
        ax.plot((0,self.film_dose.shape[1]),(self.film_dose.center.y,self.film_dose.center.y),'k--')
        ax.set_xlim(0, self.film_dose.shape[1])
        ax.set_ylim(self.film_dose.shape[0],0)
        ax.set_title(msg)
        print(msg)
        plt.cursor = Cursor(ax, useblit=True, color='white', linewidth=1)
        
        self.fig.canvas.mpl_connect('button_press_event', self.onclick_norm)
        self.cid = self.fig.canvas.mpl_connect('key_press_event', self.apply_factor_from_roi_press_enter)         
        self.wait = True
        while self.wait: plt.pause(1)
        self.cleanup()
            
    def onclick_norm(self, event):
        ax = plt.gca()
        if event.dblclick:
            size_px = self.norm_roi_size * self.film_dose.dpmm / 2
            self.roi_center = ([int(event.xdata), int(event.ydata)])
            self.roi_xmin, self.roi_xmax = int(event.xdata - size_px), int(event.xdata + size_px)
            self.roi_ymin, self.roi_ymax = int(event.ydata - size_px), int(event.ydata + size_px)
            
            rect = plt.Rectangle( (min(self.roi_xmin,self.roi_xmax),min(self.roi_ymin,self.roi_ymax)), np.abs(self.roi_xmin-self.roi_xmax), np.abs(self.roi_ymin-self.roi_ymax), fill=False )
            ax.add_patch(rect)    
            ax.plot((self.roi_center[0]-size_px,self.roi_center[0]+size_px),(self.roi_center[1],self.roi_center[1]),'w', linewidth=2)
            ax.plot((self.roi_center[0],self.roi_center[0]),(self.roi_center[1]-size_px,self.roi_center[1]+size_px),'w', linewidth=2)
            plt.gcf().canvas.draw_idle()

    def crop_film(self):
        """  Brings up an interactive plot, where the user must define 
             a rectangle ROI that will be used to crop the film.
        """     
        msg = '\nCrop film: Click and drag to draw an ROI. Press ''enter'' when finished.'
        
        self.fig = plt.figure()
        ax = plt.gca()  
        self.film_dose.plot(ax=ax)  
        ax.plot((0,self.film_dose.shape[1]),(self.film_dose.center.y,self.film_dose.center.y),'k--')
        ax.set_xlim(0, self.film_dose.shape[1])
        ax.set_ylim(self.film_dose.shape[0],0)
        ax.set_title(msg)
        print(msg)
        
        def select_box(eclick, erelease):
            x1, y1 = int(eclick.xdata), int(eclick.ydata)
            x2, y2 = int(erelease.xdata), int(erelease.ydata)   
            self.roi_xmin, self.roi_xmax = min(x1,x2), max(x1,x2)
            self.roi_ymin, self.roi_ymax = min(y1,y2), max(y1,y2)

        self.rs = RectangleSelector(ax, select_box, useblit=True, button=[1], minspanx=5, minspany=5, spancoords='pixels', interactive=True)  
        self.cid = self.fig.canvas.mpl_connect('key_press_event', self.crop_film_press_enter)
        self.wait = True
        while self.wait: plt.pause(1)
        self.cleanup()
        
    def crop_film_press_enter(self, event):
        """ Function called from crop_film() when ''enter'' is pressed. """      
        if event.key == 'enter':           
            left = self.roi_xmin
            right = self.film_dose.shape[1] - self.roi_xmax
            top = self.roi_ymin
            bottom = self.film_dose.shape[0] - self.roi_ymax
            self.film_dose.crop(left,'left')
            self.film_dose.crop(right,'right')
            self.film_dose.crop(top,'top')
            self.film_dose.crop(bottom,'bottom')  
            self.wait = False
        
    def gamma_analysis(self, film_filt=0, doseTA=3.0, distTA=3.0, threshold=0.1, norm_val='max', local_gamma=False, max_gamma=None, random_subset=None):
        """ 
        Perform Gamma analysis between registered film_dose and ref_dose.
        Gamma computation is performed using pymedphys.gamma.
    
        Parameters
        ----------
        film_filt : int, optional, default=0
            Kernel size of median filter to apply to film dose before performing gamma analysis (for noise reduction).
    
        doseTA : float, optional, default=3.0
            Dose to agreement threshold [%].
    
        distTA : float, optional, default=3.0
            Distance to agreement threshold [mm].
    
        threshold : float, optional, default=0.1
            The percent lower dose cutoff below which gamma will not be calculated. Must be between 0 and 1.
    
        norm_val : float or 'max', optional, default='max'
            Normalisation value [cGy] of reference dose, used to calculate the dose to agreement threshold and lower dose threshold.
            If 'max', the maximum dose from the reference distribution will be used.
    
        local_gamma : bool, optional, default=False
            Whether or not local gamma should be used instead of global.
    
        max_gamma : float, optional, default=None
            The maximum gamma searched for. This can be used to speed up calculation.
            Once a search distance is reached that would give gamma values larger than this parameter, the search stops.
    
        random_subset : float, optional, default=None
            Used to only calculate a random subset fraction of the reference grid, to speed up calculation. Must be between 0 and 1.
        """
        self.doseTA, self.distTA = doseTA, distTA
        self.film_filt, self.threshold, self.norm_val = film_filt, threshold, norm_val        
        start_time = time.time()
        self.GammaMap = self.computeGamma(doseTA=doseTA, distTA=distTA, threshold=threshold, norm_val=norm_val, local_gamma=local_gamma, max_gamma=max_gamma, random_subset=random_subset)       
        print(f"--- Done! ({time.time() - start_time:.1f} seconds) ---")
        self.computeDiff()
    
    def computeHDmedianDiff(self, threshold=0.8, ref = 'max'):
        """
        Compute median difference between film and reference doses in the high dose region.
    
        Parameters
        ----------
        threshold : float, optional, default=0.8
            The relative threshold (with respect to 'ref') used to determine the high dose region.
            Must be between 0 and 1.
    
        ref : 'max' or float, optional, default='max'
            The dose [cGy] used as a reference for the threshold.
            If 'max', the maximum dose in ref_dose will be used.
        """
        if ref == 'max': HDthreshold = threshold * self.ref_dose.array.max()
        else:  HDthreshold = threshold * ref
        film_HD = self.film_dose.array[self.ref_dose.array > HDthreshold]
        ref_HD = self.ref_dose.array[self.ref_dose.array > HDthreshold]
        self.HD_median_diff = np.median((film_HD-ref_HD)/ref_HD) * 100
        return self.HD_median_diff
            
    def computeDiff(self):
        """ Compute the difference map with the reference image. """
        self.DiffMap = ArrayImage(self.film_dose.array - self.ref_dose.array, dpi=self.film_dose.dpi)
        self.RelError = ArrayImage(100*(self.film_dose.array - self.ref_dose.array)/self.ref_dose.array, dpi=self.film_dose.dpi)
        self.DiffMap.MSE = sum(sum(self.DiffMap.array**2)) / len(self.film_dose.array[(self.film_dose.array > 0)]) 
        self.DiffMap.RMSE = np.sqrt(self.DiffMap.MSE)    
    
    def computeGamma(self, doseTA=2, distTA=2, threshold=0.1, norm_val=None, local_gamma=False, max_gamma=None, random_subset=None):
        """
        Compute Gamma (using pymedphys.gamma).
    
        Parameters
        ----------
        doseTA : float, optional, default=2
            Dose to agreement threshold [%].
    
        distTA : float, optional, default=2
            Distance to agreement threshold [mm].
    
        threshold : float, optional, default=0.1
            The percent lower dose cutoff below which gamma will not be calculated.
            Must be between 0 and 1.
    
        norm_val : float or None, optional, default=None
            Normalization value [cGy] of the reference dose, used to calculate the dose to agreement threshold
            and lower dose threshold. If None, no normalization is applied.
    
        local_gamma : bool, optional, default=False
            Whether to use local gamma instead of global gamma.
    
        max_gamma : float or None, optional, default=None
            The maximum gamma value to search for. Can speed up calculation by stopping the search once
            gamma values exceed this parameter.
    
        random_subset : float or None, optional, default=None
            If set, calculates gamma for only a random subset fraction of the reference grid to speed up calculation.
            Must be between 0 and 1.
    
        Returns
        -------
        ArrayImage
            The computed gamma map as an ArrayImage object.
        """
        print(f"\nComputing {doseTA}%/{distTA} mm Gamma...")
        # error checking
        if not is_close(self.film_dose.dpi, self.ref_dose.dpi, delta=3):
            raise AttributeError(f"The image DPIs to not match: {self.film_dose.dpi:.2f} vs. {self.ref_dose.dpi:.2f}")
        same_x = is_close(self.film_dose.shape[1], self.ref_dose.shape[1], delta=1.1)
        same_y = is_close(self.film_dose.shape[0], self.ref_dose.shape[0], delta=1.1)
        if not (same_x and same_y):
            raise AttributeError(f"The images are not the same size: {self.film_dose.shape} vs. {self.ref_dose.shape}")

        # set up reference and comparison images
        film_dose, ref_dose = ArrayImage(copy.copy(self.film_dose.array)), ArrayImage(copy.copy(self.ref_dose.array))
        
        if self.film_filt:
            film_dose.array = medfilt(film_dose.array, kernel_size=(self.film_filt, self.film_filt))

        if norm_val is not None:
            if norm_val == 'max': norm_val = ref_dose.array.max()
            film_dose.normalize(norm_val)
            ref_dose.normalize(norm_val)
            
        # set coordinates [mm]
        # x_coord = (np.array(range(0, self.ref_dose.shape[0])) / self.ref_dose.dpmm - self.ref_dose.physical_shape[0]/2).tolist()
        # y_coord = (np.array(range(0, self.ref_dose.shape[1])) / self.ref_dose.dpmm - self.ref_dose.physical_shape[1]/2).tolist()
        x_coord = (np.array(range(0, self.ref_dose.shape[0])) / self.ref_dose.dpmm).tolist()
        y_coord = (np.array(range(0, self.ref_dose.shape[1])) / self.ref_dose.dpmm).tolist()
        axes_reference, axes_evaluation = (x_coord, y_coord), (x_coord, y_coord)
        dose_reference, dose_evaluation = ref_dose.array, film_dose.array

        # set film_dose = 0 to Nan to avoid computing on padded pixels
        dose_evaluation[dose_evaluation == 0] = 'nan'
        
        # Compute the number of pixels to analyze
        if random_subset: random_subset = int(len(dose_reference[dose_reference >= threshold].flat) * random_subset)
        
        # Gamma computation and set maps
        gamma = pymedphys.gamma(axes_reference, dose_reference, axes_evaluation, dose_evaluation, doseTA, distTA, threshold*100,
                                local_gamma=local_gamma, interp_fraction=10, max_gamma=max_gamma, random_subset=random_subset)
        
        GammaMap = ArrayImage(gamma, dpi=film_dose.dpi)
        GammaMap.fail = ArrayImage((GammaMap.array > 1.0).astype(int), dpi=film_dose.dpi)
        GammaMap.passed = ArrayImage((GammaMap.array <= 1.0).astype(int), dpi=film_dose.dpi)
        GammaMap.npassed = np.sum(GammaMap.passed.array == 1)
        GammaMap.nfail = np.sum(GammaMap.fail.array == 1)
        GammaMap.npixel = GammaMap.npassed + GammaMap.nfail
        GammaMap.passRate = GammaMap.npassed / GammaMap.npixel * 100
        GammaMap.mean = np.nanmean(GammaMap.array)
        
        return GammaMap
                    
    def plot_gamma_var(self, param, ax=None, start=0.5, stop=4, step=0.5, varTA = None):
        """
        Plot Gamma pass rate as a function of a varying parameter (either doseTA or distTA).
    
        Parameters
        ----------
        param : str
            The parameter to vary, either 'DoseTA' (dose to agreement threshold) or 'DistTA' (distance to agreement threshold).
    
        ax : matplotlib.pyplot.Axes, optional, default=None
            Axis in which to plot the graph.
            If None, a new plot is created.
    
        start : float, optional, default=0.5
            Minimum value of the parameter to vary.
    
        stop : float, optional, default=4.0
            Maximum value of the parameter to vary.
    
        step : float, optional, default=0.5
            Increment of the parameter value between start and stop.
            
        varTA : float, optional
            Value for dose or distance used in the variable dose or distance gamma analysis (based on param)
            Default is None (which will take value used in previous gamma analysis)
        """
        values = np.arange(start, stop + step, step)        # Include stop endpoint
        GammaVar = np.zeros((len(values), 2))
        if param == 'DoseTA':
            if not varTA: varTA = self.distTA
            title = f'Variable DoseTA, DistTA = {varTA} mm'
        elif param == 'DistTA':
            if not varTA: varTA = self.doseTA
            title = f'Variable DistTA, DoseTA = {varTA} %'
    
        for i, value in enumerate(values):
            if param == 'DoseTA': gamma = self.computeGamma(doseTA=value, distTA=varTA, threshold=self.threshold, norm_val=self.norm_val)
            elif param == 'DistTA': gamma = self.computeGamma(doseTA=varTA, distTA=value, threshold=self.threshold, norm_val=self.norm_val)
            GammaVar[i] = [value, gamma.passRate]
    
        if ax is None:
            fig, ax = plt.subplots()
        x, y = GammaVar[:, 0], GammaVar[:, 1]
        ax.plot(x, y, 'o-')
        ax.set_title(title)
        ax.set_xlabel(f'{param} (%)' if param == 'DoseTA' else f'{param} (mm)')
        ax.set_ylabel('Gamma pass rate (%)')

  
    def plot_gamma_varDoseTA(self, ax=None, start=0.5, stop=4, step=0.5, varTA = None):
        """ Plot graph of Gamma pass rate vs variable doseTA.
            Note: values of distTA, threshold and norm_val will be taken as those 
            from the previous "standard" gamma analysis.
            
            Parameters
            ----------
            ax : matplotlib.pyplot.Axes, optional, default=None
                Axis in which to plot the graph.
                If None, a new plot is made.
                
            start : float, optional, default=0.5
                Minimum value of dose to agreement threshold [%]

            stop : float, optional, default=4.0
                Maximum value of dose to agreement threshold [%]

            step : float, optional, default=0.5
                Increment of dose to agreement value between start and stop values [%]
                
            varTA : float, optional
                Value for distance used in the variable dose gamma analysis
                Default is None (which will take value used in previous gamma analysis)
        """
        self.plot_gamma_var('DoseTA', ax, start, stop, step, varTA)
        
    def plot_gamma_varDistTA(self, ax=None, start=0.5, stop=4, step=0.5, varTA = None):
        """ Plot graph of Gamma pass rate vs variable distTA
            Note: values of doseTA, threshold and norm_val will be taken as those 
            from the previous "standard" gamma analysis.
            
            Parameters
            ----------
            ax : matplotlib.pyplot.Axes, optional, default=None
                Axis in which to plot the graph.
                If None, a new plot is made.
                
            start : float, optional, default=0.5
                Minimum value of dist to agreement threshold [mm]

            stop : float, optional, default=4.0
                Maximum value of dist to agreement threshold [mm]

            step : float, optional, default=0.5
                Increment of dist to agreement value between start and stop values [mm]
                
            varTA : float, optional
                Value for dose used in the variable distance gamma analysis
                Default is None (which will take value used in previous gamma analysis)
        """
        self.plot_gamma_var('DistTA', ax, start, stop, step, varTA)  

    def plot_gamma_hist(self, ax=None, bins='auto', range=[0,3]):
        """ Plot a histogram of gamma map values.

            Parameters
            ----------
            ax : matplotlib.pyplot axe object, optional, default=None
                Axis in which to plot the graph.
                If None, a new plot is made.
   
            bins : int, sequence, or str, optional, default='auto'
                Determines the number of bins in the histogram.
        
            range : tuple or None, optional, default=[0, 3]
                Determines the range of values shown in the histogram.
        """

        if ax is None:
            fig, ax = plt.subplots()
        ax.hist(self.GammaMap.array[np.isfinite(self.GammaMap.array)], bins=bins, range=range)
        ax.set_xlabel('Gamma value')
        ax.set_ylabel('Pixels count')
        ax.set_title("Gamma map histogram")
        
    def plot_gamma_pass_hist(self, ax=None, bin_size = 50):
        """ Plot a histogram of gamma map pass rate vs dose.

            Parameters
            ----------
            ax : matplotlib.pyplot axe object, optional, default=None
                Axis in which to plot the graph.
                If None, a new plot is made.

            bin_size : float, optional, default=50
                Determines the size of bins in the histogram [cGy].
                The number of bins is determined from the maximum dose in reference dose, and the bin_size.
        """

        if ax is None:
            fig, ax = plt.subplots()
        analyzed = np.isfinite(self.GammaMap.array)
        bins = np.arange(0, self.ref_dose.array.max()+bin_size, bin_size)
        dose = self.ref_dose.array[analyzed]
        gamma_pass = self.GammaMap.passed.array[analyzed]   # analyzed array includes failing gamma points
        dose_pass = (gamma_pass * dose)
        dose_pass = dose_pass[dose_pass > 0]     # Remove failing gamma points (value 0 from self.GammaMap.passed.array)
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
        ax.set_xticks(bins)
        
    def show_gamma_stats(self, figsize=(10, 10), show_hist=True, show_pass_hist=True, show_varDistTA=False, show_varDoseTA=False):
        """ Displays a figure with 4 subplots showing gamma analysis statistics:
        
            1. Gamma map histogram
            2. Gamma pass rate vs dose histogram
            3. Gamma pass rate vs variable distance to agreement threshold
            4. Gamma pass rate vs variable dose to agreement threshold
        
            Parameters
            ----------
            figsize : tuple of (float, float), optional, default=(10, 10)
                Width and height of the figure in inches.
        
            show_hist : bool, optional, default=True
                Whether to display the gamma map histogram subplot.
        
            show_pass_hist : bool, optional, default=True
                Whether to display the gamma pass rate vs dose histogram subplot.
        
            show_varDistTA : bool, optional, default=False
                Whether to display the gamma pass rate vs variable distance to agreement threshold subplot.
        
            show_varDoseTA : bool, optional, default=False
                Whether to display the gamma pass rate vs variable dose to agreement threshold subplot.
        """

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        ax_iter = iter(axes.flatten())
        
        if show_hist:      self.plot_gamma_hist(ax=next(ax_iter))
        if show_pass_hist: self.plot_gamma_pass_hist(ax=next(ax_iter))
        if show_varDistTA: self.plot_gamma_varDistTA(ax=next(ax_iter))
        if show_varDoseTA: self.plot_gamma_varDoseTA(ax=next(ax_iter))
        plt.tight_layout()
        plt.show()
        
    def plot_profile(self, ax=None, profile='x', position=None, title=None, diff=False, offset=0, vertical_line=None, xlim=None, ylim='auto'):
        """ Plot a line profile of reference dose and film dose at a given position.
        
            Parameters
            ----------
            ax : matplotlib.pyplot.Axes, optional, default=None
                Axis in which to plot the graph.
                If None, a new plot is made.
        
            profile : {'x', 'y'}, optional, default='x'
                The orientation of the profile to plot (x: horizontal, y: vertical).
        
            position : int, optional, default=None
                The position of the profile to plot, in pixels, in the direction perpendicular to the profile.
                For example, if profile='x' and position=400, a profile in the x direction is shown at position y=400.
                If None, position is set to the center of the reference dose.
        
            title : str, optional, default=None
                The title to display on the graph.
                If None, the title is set automatically to display profile direction and position.
        
            diff : bool, optional, default=False
                If True, the difference in profiles (film - reference) is displayed.
        
            offset : int, optional, default=0
                If a known offset exists between the film and the reference dose, the plotted profile can be shifted
                to account for this offset. For example, a film exposed at a fixed gantry angle could have a known 
                offset due to gantry sag, and you may want to correct for it on the profile.
        
            vertical_line : int, optional, default=None
                If set, a dashed vertical line is plotted on the profile at this position.
        
            xlim : tuple, optional, default=None
                If given, xlim will be passed to ax.set_xlim(xlim) to set the x-axis limits.
        
            ylim : tuple, 'max', or 'auto', optional, default='auto'
                If given a tuple, ylim will be passed to ax.set_ylim(ylim) to set the y-axis limits.
                If 'max', ylim goes from 0 to 105% of the maximum reference dose.
                If 'auto', ylim goes from 0 to 105% of the maximum of either the current reference or film dose profile.
        """     

        film, ref = self.film_dose.array, self.ref_dose.array

        if profile == 'x':
            if position is None: position = int(self.film_dose.center.y)
            film_prof, ref_prof = film[position,:], ref[position,:] 
        elif profile == 'y':
            if position is None: position = int(self.film_dose.center.x)
            film_prof, ref_prof = film[:,position], ref[:,position]
        
        x_axis = (np.array(range(0, len(film_prof))) / self.film_dose.dpmm).tolist()

        if ax is None: fig, ax = plt.subplots()    
        ax.clear()
        ax.plot([i+offset for i in x_axis], film_prof, 'r-', linewidth=2, label='Film')
        ax.plot(x_axis, ref_prof, 'b--', linewidth=2, label='Reference')
        
        if title is None:
            if profile == 'x': title=f'Horizontal Profile (y = {int(position / self.film_dose.dpmm)} mm)'
            if profile == 'y': title=f'Vertical Profile (x = {int(position / self.film_dose.dpmm)} mm)'
        ax.set_title(title)
        ax.set_xlabel('Position (mm)')
        ax.set_ylabel('Dose (cGy)')
        
        if diff:
            ax_diff = ax.twinx()
            ax_diff.set_ylabel("Difference (cGy)")
            ax_diff.plot(x_axis, film_prof - ref_prof,'g-', linewidth=0.25)
            
        if xlim: ax.set_xlim(xlim)
        if ylim == 'max': ax.set_ylim((0, self.ref_dose.array.max() * 1.05))
        elif ylim == 'auto': ax.set_ylim((0, max(np.concatenate((film_prof, ref_prof))) * 1.05))
        else: ax.set_ylim(ylim)
            
        if vertical_line:
            ax.axvline(x=vertical_line / self.film_dose.dpmm, color='k', linestyle=':', linewidth=1)
        
    def show_isodoses(self, ax=None, levels=None, colors=None, show_ruler=True, figsize=(15,15)):
        """ Display isodose lines for the film dose and reference dose on a given axis.
            
            Parameters
            ----------
            ax : matplotlib.pyplot.Axes, optional, default=None
                Axis in which to plot the isodose lines.
                If None, a new figure and axis are created.
            
            levels : list of float, optional, default=None
                Specific dose levels to plot as isodose lines.
                If None, levels are set to 20%, 40%, 60%, and 80% of the maximum reference dose.
            
            colors : list of str or matplotlib.colors, optional, default=None
                Colors to use for the isodose lines.
                If None, a default colormap is used.
            
            show_ruler : bool, optional, default=True
                Whether to add an interactive ruler to measure distances on the plot using the mouse.
            
            figsize : tuple of (float, float), optional, default=(15, 15)
                Width and height of the figure in inches if a new figure is created.            
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        if levels is None:
            levels = [self.ref_dose.array.max() * l for l in np.arange(0.2, 1.0, 0.2)]
        if colors is None:
            colors = plt.cm.tab10(np.linspace(0, 1, 10))
        extent = [0, self.ref_dose.physical_shape[1], self.ref_dose.physical_shape[0], 0]
        self.film_dose.plot_isodoses(ax=ax, levels=levels, colors=colors, linestyles='dashdot', linewidths=0.5, extent=extent, inline=False)
        self.ref_dose.plot_isodoses(ax=ax, levels=levels, colors=colors, linestyles='solid', linewidths=0.5, labels=False, extent=extent)
        legend_lines = [plt.Line2D([0], [0], linestyle='dotted', color='black', label='Film Dose'),
                        plt.Line2D([0], [0], linestyle='solid', color='black', label='Reference Dose')]
        plt.legend(handles=legend_lines)
        ax.invert_yaxis()
        if show_ruler:
            self.ruler = add_ruler(ax)

    def show_results(self, fig=None, x=None, y=None, show=True):
        """ Display an interactive figure showing the results of a gamma analysis.
            The figure contains 6 axis, which are, from left to right and top to bottom:
            Film dose, reference dose, gamma map, relative error, x profile and y profile.
            
            Parameters
            ----------
            fig : matplotlib.pyplot figure object, optional, default=None
                Figure in which to plot the graph.
                If None, a new figure is made.
            
            x, y : int, optional, default=None
                Initial x/y coordinates of the profiles.
                If None, profile will be at image center.
        """
        a = None
        
        if x is None: self.prof_x = self.ref_dose.shape[1] // 2
        elif x == 'max':
            a = np.unravel_index(self.ref_dose.array.argmax(), self.ref_dose.array.shape)
            self.prof_x = a[1]
        else: self.prof_x = x
        if y is None: self.prof_y = self.ref_dose.shape[0] // 2
        elif y == 'max':
            if a is None: a = np.unravel_index(self.ref_dose.array.argmax(), self.ref_dose.array.shape)
            self.prof_y = a[0]
        else: self.prof_y = y
         
        fig, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2, figsize=(10, 8))
        fig.tight_layout()
        axes = [ax1,ax2,ax3,ax4,ax5,ax6]
        for ax in axes[0:4]:                # Share x/y axes for zoom purposes.
            ax.sharex(axes[0])
            ax.sharey(axes[0])
        fig.canvas.manager.set_window_title(f"Facteur{self.film_dose_factor:.2f}_Filtre{self.film_filt}_Gamma{self.doseTA}%-{self.distTA}mm")
        clim = [0, np.percentile(self.ref_dose.array, 99.9).round(-1)]

        self.film_dose.plot(ax1, clim=clim, title=f'Film Dose ({os.path.basename(self.film_dose.path)})', colorbar=True)
        self.ref_dose.plot(ax2, clim=clim, title=f'Reference Dose ({os.path.basename(self.ref_dose.path)})', colorbar=True)
        self.GammaMap.plot(ax3, clim=[0,2], cmap='bwr', title=f'Gamma Map ({self.GammaMap.passRate:.2f}% Pass; {self.GammaMap.mean:.2f} Mean)', colorbar=True)
        ax3.set_facecolor('k')
        min_value = max(-20, np.percentile(self.DiffMap.array,[1])[0].round(decimals=0))
        max_value = min(20, np.percentile(self.DiffMap.array,[99])[0].round(decimals=0))
        clim = [min_value, max_value]    
        self.RelError.plot(ax4, cmap='jet', clim=clim, title=f'Relative Error (%) (RMSE = {self.DiffMap.RMSE:.2f})', colorbar=True)
        self.show_profiles(axes, x=self.prof_x, y=self.prof_y)
        plt.multi = MultiCursor(None, (axes[0],axes[1],axes[2],axes[3]), color='r', lw=1, horizOn=True)
        
        fig.canvas.mpl_connect('button_press_event', lambda event: self.set_profile(event, axes))
        fig.canvas.mpl_connect('key_press_event', self.show_results_ontype)
        if show: plt.show()
        
    def show_results_ontype(self, event):
        if event.key == 'enter':
            self.get_profile_offsets(x=self.prof_x, y=self.prof_y)
        
    def show_profiles(self, axes, x, y, figsize=(10,10)):
        """ This function is called by show_results and set_profile to draw dose profiles
            at a given x/y coordinates, and draw lines on the dose distribution maps
            to show where the profile is taken.
        """
        ax_x, ax_y = axes[-2], axes[-1]
        self.plot_profile(ax=ax_x, profile='x', position=y, vertical_line=x)
        self.plot_profile(ax=ax_y, profile='y', position=x, vertical_line=y)
        
        for ax in axes[:4]:
            for line in ax.lines: line.remove()
            ax.plot([x, x], [0, self.ref_dose.shape[0]], 'w--', linewidth=1)
            ax.plot([0, self.ref_dose.shape[1]], [y, y], 'w--', linewidth=1)
        
    def set_profile(self, event, axes):
        """ This function is called by show_results to draw dose profiles
            on mouse click (if cursor is not set to zoom or pan).
        """
        if event.button == 1 and plt.gcf().canvas.cursor().shape() == 0:   # 0 is the arrow, which means we are not zooming or panning.
            if event.inaxes in axes[0:4]:
                self.prof_x = int(event.xdata)
                self.prof_y = int(event.ydata)
            elif event.inaxes == axes[4]: self.prof_x = int(event.xdata * self.film_dose.dpmm)
            elif event.inaxes == axes[5]: self.prof_y = int(event.xdata * self.film_dose.dpmm)
            
            self.show_profiles(axes,x=self.prof_x, y=self.prof_y)    
            plt.gcf().canvas.draw_idle()
        else: print('\nZoom/pan is currently selected.\nNote: Unable to set profile when this tool is active.')
        
        
    #=================== Registration functions ======================
    def register(self, shift_x=0, shift_y=0, rot=0, threshold=10, register_using_gradient=False, markers_center=None):
        """ Starts the registration procedure between film and reference dose.
            
            Parameters
            ----------
            shift_x / shift_y : float, optional, default=0
                Apply a shift [mm] in the x/y direction between reference dose and film dose. 
                Used if there is a known shift between the registration point in the reference image and the film image.
                
            rot : float, optional, default=0
                Apply a known rotation [degrees] between reference dose and film dose. 
                Used if the markers on the reference image are known to be not perfectly aligned
                in an horizontal/vertical line.
            
            threshold : int, optional, default=10
                Threshold value [cGy] used in detecting film edges for auto-cropping.
            
            register_using_gradient : bool, optional, default=False
                Determine if the registration results (overlay of film/ref dose) will be displayed 
                after applying a sobel filter to improve visibility of sharp dose gradients.
            
            markers_center : list of 3 floats, optional, default=None
                Coordinates [mm] in the reference dose corresponding to the marks intersection on the film (R-L, I-S, P-A).
                It will be used to align the reference point on the film (given by the intersection of the two lines
                determined by the four marks made on the edges of the film) to an absolute position in the reference dose.
                If None, the film reference point will be positioned to the center of the reference dose.            
        """
        self.register_using_gradient = register_using_gradient
        self.shifts = [shift_x, shift_y]
        self.rot = rot
        self.markers_center = markers_center
        if threshold > 0 :
            self.film_dose.crop_edges(threshold=threshold)
        
        self.film_dose.plot()
        self.select_markers()
        self.tune_registration()
        
    def select_markers(self):
        """ Start the interactive plot where the 4 markers on the film must be identified. """
        self.fig, self.ax = plt.gcf(), plt.gca()
        self.markers = []
        
        print('\nPlease double-click on each marker. Press "Enter" when done')
        print('Keyboard shortcuts: Numpad arrows: Move last placed marker; r = Rotate 90 degrees; h = Flip horizontally; v = Flip vertically')
        self.ax.set_title('Marker 1 = ; Marker 2 = ; Marker 3 = ; Marker 4 = ')
        
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.canvas.mpl_connect('key_press_event', self.ontype)
        self.fig.canvas.mpl_connect('close_event', self.onclose)
        self.cursor = Cursor(self.ax, useblit=True, color='white', linewidth=1)
        plt.show()
        
        self.wait = True
        while self.wait and plt.fignum_exists(self.fig.number): plt.pause(1)
        self.cleanup()
        
    def cleanup(self):
        if hasattr(self, "rs"): del self.rs    
        if hasattr(self, "cursor"): del self.cursor    
        if self.fig:
            self.fig.canvas.mpl_disconnect(self.cid)
            plt.close(self.fig)
    
    def onclose(self, event):
        """ Handle the figure close event. """
        self.wait = False
    
    def onclick(self, event):
        """ Set the markers' coordinates when the mouse is double-clicked. """
        if event.dblclick and len(self.markers) < 4: 
            self.markers.append([int(event.xdata), int(event.ydata)])
            self.plot_markers()            

    def plot_markers(self):      
        """ Plot the markers on the figure. """
        l = 20  # Length of crosshair/marker
        
        for m in self.markers:
            self.ax.plot((m[0] - l, m[0] + l), (m[1], m[1]), 'w', linewidth=1)
            self.ax.plot((m[0], m[0]), (m[1] - l, m[1] + l), 'w', linewidth=1)
        
        marker_titles = [f"Marker {i + 1} = {m}" for i, m in enumerate(self.markers)]
        title = "; ".join(marker_titles) + " ; " * (4 - len(self.markers))
        self.ax.set_title(title)
        self.fig.canvas.draw_idle()
        
    def ontype(self, event):
        """ Handle keyboard events for rotation, flipping, and marker movement. """
        
        def reset_markers(reason = "change"):
            """ Resets self.markers and updates the marker text/title in the figure. """
            if reason == "change": print('\nFilm dose array has updated...')
            elif reason == "less": print(f'\n{len(self.markers)} markers were selected when 4 were expected...')
            print('Please start over...')
            print('Please double-click on each marker. Press ''enter'' when done')
            self.markers = []
            self.ax.set_title('Marker 1 = ; Marker 2 = ; Marker 3 = ; Marker 4 = ')
            self.fig.canvas.draw_idle()    
            
        def update_markers():
            """ Updates the markers on the plot. """
            for line in self.ax.lines: line.remove()
            self.plot_markers()
        
        # Handle key actions for rotation and flipping
        key_actions = {
            'r': lambda: setattr(self.film_dose, 'array', np.rot90(self.film_dose.array, k=1)),
            'l': lambda: setattr(self.film_dose, 'array', np.fliplr(self.film_dose.array)),
            'v': lambda: setattr(self.film_dose, 'array', np.flipud(self.film_dose.array))
        }
        
        if event.key in key_actions:
            self.ax.clear()
            key_actions[event.key]()  # Apply the respective transformation
            reset_markers()
            self.fig.canvas.draw_idle()
            self.film_dose.plot(ax=self.ax)
            return
                
        # Handle marker movement
        direction_map = {
            '8': (0, -1),  # Move up
            '2': (0, 1),   # Move down
            '4': (-1, 0),  # Move left
            '6': (1, 0)    # Move right
        }
        
        if len(self.markers) > 0 and event.key in direction_map:
            dx, dy = direction_map[event.key]
            self.markers[-1][0] += dx
            self.markers[-1][1] += dy
            update_markers()
            return
            
        if event.key == 'enter':
            if len(self.markers) == 0:
                max_x, max_y = self.film_dose.array.shape[1], self.film_dose.array.shape[0]
                self.markers = [[max_x/2, 0], [max_x, max_y/2], [max_x/2, max_y], [0, max_y/2]]
                print("\nNo markers selected. \nCenter of film dose array selected for markers. \nAdjust registration as needed.")
            elif len(self.markers) != 4:
                self.film_dose.plot(ax=self.ax)
                reset_markers("less")
            else:
                print(f"Marker 1: {self.markers[0]}; Marker 2: {self.markers[1]}; Marker 3: {self.markers[2]}; Marker 4: {self.markers[3]}.")
                self.move_iso_center()
                self.remove_rotation()
                if self.ref_dose is not None: self.apply_shifts_ref()
                if self.rot: self.film_dose.rotate(self.rot)
                self.wait = False
                
    def move_iso_center(self):
        """ Register the film dose and reference dose by moving the reference
            point to the center of the image (by padding).
            The reference point is given by the intersection of the two lines
            connecting the two markers on opposite side of the film, and
            by absolute coordinates in the stored in self.markers_center
            for the reference dose.
        """
        
        # Find the indices of markers on top, bottom, left, right of the film.
        x, y = [m[0] for m in self.markers], [m[1] for m in self.markers]
        t, b, l, r = y.index(min(y)), y.index(max(y)), x.index(min(x)), x.index(max(x))
        
        # Find intersection of the lines top-bottom and left-right and set the reference point (x0, y0).
        line1, line2 = ((x[t],y[t]),(x[b],y[b])), ((x[r],y[r]),(x[l],y[l]))
        (x0,y0) = line_intersection(line1, line2)    
        self.x0, self.y0 = int(np.around(x0)), int(np.around(y0))
        
        # Make (x0, y0) the center of image by padding
        self.film_dose.move_pixel_to_center(x0, y0) 
        
        # Move the reference point in the reference dose to the center
        # NOTE: This section is made to work with planar dose exported from RayStation
        # in DICOM format. It will probably need to be changed if you use a different TPS.
        markers_center = self.markers_center
        if markers_center is not None:
            pos = [float(i) for i in self.ref_dose.metadata.ImagePositionPatient]
            sizeX, sizeY = self.ref_dose.metadata.Columns, self.ref_dose.metadata.Rows
            orientation = self.ref_dose.metadata.SeriesDescription
            dpmm = self.ref_dose.dpmm
            
            if 'Transversal' in orientation:
                x_corner, y_corner = pos[0], -pos[1]
                x_marker, y_marker = markers_center[0], markers_center[2]
            elif 'Sagittal' in orientation:
                x_corner, y_corner = -pos[1], pos[2]
                x_marker, y_marker = markers_center[2], markers_center[1]
            elif 'Coronal' in orientation:
                x_corner, y_corner = pos[0], pos[2]
                x_marker, y_marker = markers_center[0], markers_center[1]
            else:
                raise ValueError("Unsupported orientation")

            x_pos_mm, y_pos_mm = x_marker - x_corner, y_marker - y_corner
            
            x0 = int(np.around(x_pos_mm * dpmm))
            if 'Sagittal' in orientation:
                x0 = sizeX + int(np.around(x_pos_mm * dpmm))
            
            y0 = sizeY - int(np.around(y_pos_mm * dpmm))
            if 'Transversal' in orientation:
                y0 = int(np.around(y_pos_mm * dpmm))

            self.ref_dose.move_pixel_to_center(x0, y0)
            
    def remove_rotation(self):
        """ Rotates the film around the center so that left/right
            and top/bottom markers are horizontally and vertically aligned.  
        """
        # Find the indices of markers on top, bottom, left, right of the film.
        x, y = [m[0] for m in self.markers], [m[1] for m in self.markers]
        t, b, l, r = y.index(min(y)), y.index(max(y)), x.index(min(x)), x.index(max(x))
        
        # Calculate rotation angles for vertical and horizontal alignment
        angle1 = math.degrees(math.atan2(x[b] - x[t], y[b] - y[t]))
        angle2 = math.degrees(math.atan2(y[l] - y[r], x[r] - x[l]))
        
        # Appy inverse rotation
        angle_corr = -0.5 * (angle1 + angle2)
        print(f'Applying a rotation of {angle_corr} degrees')
        self.film_dose.rotate(angle_corr)
            
    def apply_shifts_ref(self):
        """ Apply shifts given in self.shifts by padding the reference image. """
        pad_x_pixels =  int(round(self.shifts[0] * self.ref_dose.dpmm )) *2
        pad_y_pixels =  int(round(self.shifts[1] * self.ref_dose.dpmm )) *2

        # Apply padding to the reference image based on calculated pixel shifts
        if pad_x_pixels != 0:
            edge = 'left' if pad_x_pixels > 0 else 'right'
            self.ref_dose.pad(pixels=abs(pad_x_pixels), value=0, edges=edge)
        if pad_y_pixels != 0:
            edge = 'top' if pad_y_pixels > 0 else 'bottom'
            self.ref_dose.pad(pixels=abs(pad_y_pixels), value=0, edges=edge)
    
    def tune_registration(self): 
        """ Starts the registration fine tuning process.
            The registered film and reference image are displayed superposed.
            User can adjust the registration using keyboard shortcuts.
            Arrow keys will move the film dose in one pixel increments.
            Ctrl+left/right will rotate the film dose by 0.1 degrees counterclockwise/clockwise.
        """
        if self.ref_dose is None:
            self.ref_dose = self.film_dose
        film_dose_path, ref_dose_path = self.film_dose.path, self.ref_dose.path
        
        # Make the film and reference images the same size
        self.film_dose, self.ref_dose = equate_images(self.film_dose, self.ref_dose)
        self.film_dose.path, self.ref_dose.path = film_dose_path, ref_dose_path

        print('\nFine tune registration using keyboard if needed. Arrow keys = move; ctrl+left/right = rotate. Press enter when done.')
        
        self.fig, ax = plt.subplots()
        self.cid = self.fig.canvas.mpl_connect('key_press_event', self.reg_ontype)
        img_array = self.film_dose.array - self.ref_dose.array
        min_val, max_val = np.percentile(img_array, [1, 99]).round(decimals=-1)
        lim = max(abs(min_val), abs(max_val))
        self.clim = [-lim, lim]
        self.show_registration(ax=ax)

        self.wait = True
        while self.wait: plt.pause(1)
        self.cleanup()
        
    def show_registration(self, ax=None, cmap='bwr'):
        """ Show the superposition of the film and reference dose.
            If self.register_using_gradient is set to True, a Sobel filter is applied
            to both reference and film dose to increase dose gradients visibility.
        """
        if ax is None: ax = plt.gca()
        ax.clear()
        
        # Apply Sobel filter if using gradients
        if self.register_using_gradient:
            ref_grad = np.hypot(spf.sobel(self.ref_dose.as_type(np.float32), 1), spf.sobel(self.ref_dose.as_type(np.float32), 0))
            film_grad = np.hypot(spf.sobel(self.film_dose.as_type(np.float32), 1), spf.sobel(self.film_dose.as_type(np.float32), 0))
            img_array = film_grad - ref_grad
        else:
            img_array = self.film_dose.array - self.ref_dose.array
        img = load(img_array, dpi=self.film_dose.dpi) 
        
        # rmse =  (sum(sum(img.array**2)) / len(self.film_dose.array[(self.film_dose.array > 0)]))**0.5
        rmse = np.sqrt(np.mean(img_array**2))

        img.plot(ax=ax, clim=self.clim, cmap=cmap)     
        ax.plot((0, img.shape[1]), (img.center.y, img.center.y),'k--')
        ax.plot((img.center.x, img.center.x), (0, img.shape[0]),'k--')
        ax.set_xlim(0, img.shape[1])
        ax.set_ylim(img.shape[0],0)
        ax.set_title(f'Fine tune registration. Arrow keys = move; ctrl+left/right = rotate. Press enter when done. RMSE = {rmse:.2f}')
        
    def reg_ontype(self, event):
        """ Thie function is called by self.tune_registration() to apply translations
            and rotations, and to end the registration process when Enter is pressed.
        """
        fig, ax = plt.gcf(), plt.gca()
        
        def end_registration():
            """ End the registration process by disconnecting the event and stopping the wait loop. """
            self.fig.canvas.mpl_disconnect(self.cid)
            self.wait = False
            
        # Define key actions
        key_actions = {
            'up': lambda: self.film_dose.roll(direction='y', amount=-1),
            'down': lambda: self.film_dose.roll(direction='y', amount=1),
            'left': lambda: self.film_dose.roll(direction='x', amount=-1),
            'right': lambda: self.film_dose.roll(direction='x', amount=1),
            'ctrl+right': lambda: self.film_dose.rotate(-0.1),
            'ctrl+left': lambda: self.film_dose.rotate(0.1),
            'enter': lambda: end_registration()
        }
        
        # Apply action based on key event
        action = key_actions.get(event.key)
        if action:
            action()
            if event.key != 'enter':
                self.show_registration(ax=ax)
                fig.canvas.draw_idle()
            
    def save_current_figure(self, filename, **kwargs):
        """Save the analyzed image to a file.

        Parameters
        ----------
        filename : str
            The location and filename to save to.
        kwargs
            Keyword arguments are passed to plt.savefig().
        """
        fig = plt.gcf()
        fig.savefig(filename, **kwargs)
        plt.close(fig)
        
    def show_cluster_analysis(self, cluster_id=0, xlim_margin_mm=10, figsize=(10,10), levels=None):
        """
        Display the dose distribution with cluster analysis, including dose distribution,
        cluster location, isodoses, and profiles.
    
        Parameters
        ----------
        cluster_id : int, optional, default=0
            The ID of the cluster to display.
    
        xlim_margin_mm : int, optional, default=10
            The margin [mm] to add to the x-axis limits around the cluster location.
    
        figsize : tuple of (float, float), optional, default=(10, 10)
            Width and height of the figure in inches.
    
        levels : list of float, optional, default=None
            Specific dose levels to plot as isodose lines.
            If None, default levels are used.
        """
        # Get coordinates of selected cluster
        cluster = self.clusters_analysis[cluster_id]
        x, y = cluster['x_px'], cluster['y_px']
        x_mm, y_mm = cluster['x_mm'], cluster['y_mm']
        coords = self.ref_dose.clusters[cluster_id]['coords']
        coords_mm = coords / self.ref_dose.dpmm
        
        # Define plot limits with margins
        x_xlim = (min(coords_mm[:,1]) - xlim_margin_mm, max(coords_mm[:,1]) + xlim_margin_mm)
        y_xlim = (min(coords_mm[:,0]) - xlim_margin_mm, max(coords_mm[:,0]) + xlim_margin_mm)
        
        # Plot full dose distribution and cluster location
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=figsize)
        extent = [0, self.ref_dose.physical_shape[1], self.ref_dose.physical_shape[0], 0]
        self.ref_dose.plot(ax=ax1, extent=extent)
        ax1.plot((x_mm, x_mm),(0,self.ref_dose.shape[0]),'w--', linewidth=1)
        ax1.plot((0, self.ref_dose.shape[1]),(y_mm, y_mm),'w--', linewidth=1)    
        rect = plt.Rectangle((x_xlim[0], y_xlim[0]), x_xlim[1] - x_xlim[0], y_xlim[1] - y_xlim[0], linewidth=1, edgecolor='w', linestyle='--', fill=False)
        ax1.add_patch(rect)
        
        # Plot the isodoses
        self.show_isodoses(ax=ax2, levels=levels)
        ax2.set_xlim(x_xlim)
        ax2.set_ylim(y_xlim[1], y_xlim[0])
        
        # Plot profiles
        self.plot_profile(ax=ax3, profile='x', position=y, diff=True, xlim=x_xlim, vertical_line=x)
        self.plot_profile(ax=ax4, profile='y', position=x, diff=True, xlim=y_xlim, vertical_line=y)
            
    def publish_pdf(self, filename=None, author=None, unit=None, notes=None, open_file=False, x=None, y=None, plot_clusters_analysis=False, iso_levels=None, xlim_margin_mm=10, **kwargs):
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
        canvas = pdf.PylinacCanvas(filename, page_title=title, logo=Path(__file__).parent / 'OMG_Logo.png')
        canvas.add_text(text='Film infos:', location=(1, 25.5), font_size=12)
        text = [f'Film dose: {os.path.basename(self.film_dose.path)}',
                f'Film dose factor: {self.film_dose_factor:.2f}',
                f'Reference dose: {os.path.basename(self.ref_dose.path)}',
                f'Reference dose factor: {self.ref_dose_factor:.2f}',
                f'Film filter kernel: {self.film_filt}',
                f'Gamma threshold: {self.threshold}',
                f'Gamma dose-to-agreement: {self.doseTA}',
                f'Gamma distance-to-agreement: {self.distTA}',
                f'Gamma normalization: {self.norm_val}'
               ]
        canvas.add_text(text=text, location=(1, 25), font_size=10)
        data = io.BytesIO()
        self.show_results(x=x, y=y, show = False)
        self.save_current_figure(data)
        canvas.add_image(image_data=data, location=(0.5, 3), dimensions=(19, 19))

        if plot_clusters_analysis:   
            canvas.add_new_page()
            text = ['Detected dose clusters']
            canvas.add_text(text=text, location=(1, 25), font_size=10)
            data = io.BytesIO()
            self.ref_dose.plot_clusters()
            self.save_current_figure(data)
            canvas.add_image(image_data=data, location=(0.5, 0), dimensions=(20, 24))
            
            for i, cluster in enumerate(self.clusters_analysis):      
                canvas.add_new_page()
                canvas.add_text(text='Cluster analysis', location=(1, 25.5), font_size=12)
                text = [f'Cluster center: X = {cluster["x_mm"]:.1f} mm, Y = {cluster["y_mm"]:.1f} mm',
                        f'Median dose difference: {cluster["Dose diff"]:.2f} %',
                        f'Profile offset: X = {cluster["Offset x"]:.2f} mm, Y = {cluster["Offset y"]:.2f} mm',
                        f'Profile width difference: X = {cluster["Diff width x"]:.2f} mm, Y = {cluster["Diff width y"]:.2f} mm'
                       ]
                
                data = io.BytesIO()
                self.show_cluster_analysis(cluster_id=i, levels=iso_levels)
                self.save_current_figure(data)
                canvas.add_image(image_data=data, location=(0.5, 5), dimensions=(20, 20))
                canvas.add_text(text=text, location=(1, 25), font_size=10)

        canvas.add_new_page()
        canvas.add_text(text='Isodoses plot', location=(1, 25), font_size=10)
        data = io.BytesIO()
        self.show_isodoses(figsize=(10, 10), levels=iso_levels, show_ruler=False)
        self.save_current_figure(data)
        canvas.add_image(image_data=data, location=(0.5, 3), dimensions=(20, 20))
        
        canvas.add_new_page()
        data = io.BytesIO()
        self.show_gamma_stats(figsize=(10, 10))
        self.save_current_figure(data)
        canvas.add_image(image_data=data, location=(0.5, 2), dimensions=(20, 20))
        
        canvas.finish()
        if open_file: webbrowser.open(filename)       


    #=================== Clusters analysis ======================
    def analyse_clusters(self, clusters_threshold=0.6, xlim_margin_mm=10):
        """
        Analyze clusters in the reference dose and compute median dose differences.
    
        Parameters
        ----------
        clusters_threshold : float, optional, default=0.6
            Threshold value to identify clusters within the reference dose. The value should be between 0 and 1.
    
        xlim_margin_mm : int, optional, default=10
            The margin [mm] to add to the x-axis limits around the clusters.
        """
        self.clusters_analysis = []
        clusters = self.ref_dose.detect_clusters(threshold=clusters_threshold) 
        self.ref_dose.plot_clusters()
        
        fig, ax = plt.gcf(), plt.gca()

        for cluster in clusters:
            com = cluster['center_of_mass']
            mask = cluster['region_mask']
            coords = cluster['coords']
            coords_mm = coords / self.ref_dose.dpmm
            
            x, y = int(com[1]), int(com[0])
            x_mm, y_mm = x / self.ref_dose.dpmm, y / self.ref_dose.dpmm
            
            # Plot cluster center and lines
            for line in ax.lines: line.remove()
            ax.plot([x,x], [0, self.ref_dose.shape[0]], 'w--', linewidth=1)
            ax.plot([0, self.ref_dose.shape[1]], [y,y], 'w--', linewidth=1)
            fig.canvas.draw_idle()
            plt.pause(0.01)
                        
            # Calculate median doses and relative difference
            median_film_dose = np.median(self.film_dose.array[mask.astype(bool)])
            median_ref_dose = np.median(self.ref_dose.array[mask.astype(bool)])
            relative_diff = (median_film_dose - median_ref_dose) / median_ref_dose * 100
            print(f"Median film dose = {median_film_dose:.2f} cGy; median ref dose = {median_ref_dose:.2f} cGy; Relative diff = {relative_diff:.2f}%")
            
            x_xlim = (min(coords_mm[:,1]) - xlim_margin_mm, max(coords_mm[:,1]) + xlim_margin_mm)
            y_xlim = (min(coords_mm[:,0]) - xlim_margin_mm, max(coords_mm[:,0]) + xlim_margin_mm)
            
            self.get_profile_offsets(x=x, y=y, x_xlim=x_xlim, y_xlim=y_xlim)
            self.clusters_analysis.append({'x_px': x, 'y_px': y, 'x_mm': x_mm, 'y_mm': y_mm,
                                           'Dose diff': relative_diff,
                                           'Offset x': self.offset_x, 'Offset y': self.offset_y,
                                           'Diff width x': self.diff_x, 'Diff width y': self.diff_y })
        plt.close(fig)           
    
    #=================== Profile analysis ======================
    def get_profile_offsets(self, x=None, y=None, x_xlim=None, y_xlim=None):
        """
        Start an interactive process where the user can move the measured profile
        with respect to the reference profile to compute the spatial offset between the two.
        The process is repeated four times to get offsets on both sides in the x and y directions.
    
        Parameters
        ----------
        x : int, optional
            The x position of the profile to plot, in pixels.
            Defaults to the center of the reference dose.
    
        y : int, optional
            The y position of the profile to plot, in pixels.
            Defaults to the center of the reference dose.
    
        x_xlim : tuple of (float, float), optional
            X-axis limits for X profile plotting.
    
        y_xlim : tuple of (float, float), optional
            X-axis limits for Y profile plotting.
        """
        if x is None: x = self.ref_dose.shape[1] // 2
        if y is None: y = self.ref_dose.shape[0] // 2
        
        # Compute offsets for each direction and side
        self.offset_x_l = self.get_profile_offset(x=x, y=y, direction='x', side='left', xlim=x_xlim) 
        self.offset_x_r = self.get_profile_offset(x=x, y=y, direction='x', side='right', xlim=x_xlim)
        self.offset_y_l = self.get_profile_offset(x=x, y=y, direction='y', side='left', xlim=y_xlim)
        self.offset_y_r = self.get_profile_offset(x=x, y=y, direction='y', side='right', xlim=y_xlim)
        
        # Calculate average offsets and differences
        self.offset_x = -0.5*(self.offset_x_l + self.offset_x_r)
        self.offset_y = -0.5*(self.offset_y_l + self.offset_y_r)
        self.diff_x = self.offset_x_l - self.offset_x_r
        self.diff_y = self.offset_y_l - self.offset_y_r
        
        # Print results
        print(f"X: Offset = {self.offset_x:.2f} mm; Diff width = {self.diff_x:.2f} mm")
        print(f"Y: Offset = {self.offset_y:.2f} mm; Diff width = {self.diff_y:.2f} mm")

    def get_profile_offset(self, x, y, direction, side='left', xlim=None):
        """
        Open an interactive plot where the user can move the measured profile with
        respect to the reference profile to compute the spatial offset between the two.
    
        Parameters
        ----------
        x : int
            X position for the profile plot, in pixels (horizontal direction).
            
        y : int
            Y position for the profile plot, in pixels (vertical direction).
            
        direction : str
            Direction of the profile, either 'x' (horizontal) or 'y' (vertical).
            
        side : str, optional, default='left'
            Side of the profile to match, either 'left' or 'right'.
            
        xlim : tuple of (float, float), optional
            X-axis limits for profile plotting.
        """

        print(f'\nUse left/right keyboard arrows to move profile and fit on {side} side. Press Enter when done.')
        self.offset = 0
        self.direction = direction
        self.xlim = xlim
        
        title = f'{direction}: Fit profiles on {side} side'
        if direction == 'x':
            self.position, self.line = y, x
            self.plot_profile(profile='x', position=y, vertical_line=x, title=title, xlim=xlim)
        elif direction == 'y':
            self.position, self.line = x, y
            self.plot_profile(profile='y', position=x, vertical_line=y, title=title, xlim=xlim)
        else:
            raise ValueError("Direction must be 'x' or 'y'")

        self.fig = plt.gcf()
        plt.get_current_fig_manager().window.showMaximized()
        self.cid = self.fig.canvas.mpl_connect('key_press_event', self.move_profile_ontype)
        
        # Wait for user interaction
        self.wait = True
        while self.wait: plt.pause(1)
        self.cleanup()
        return self.offset
                
    def move_profile_ontype(self, event):
        """ This function is called by self.get_profile_offset()
            to either move the profile when left/right keys are pressed,
            or to close the figure when Enter is pressed.
        """
        fig, ax = plt.gcf(), plt.gca()
        
        # Update offset and plot profile based on key press
        if event.key in ['left', 'right']:
            offset = -0.1 if event.key == 'left' else 0.1
            self.offset += offset
            title = f'Shift = {self.offset:.1f} mm'
            self.plot_profile(ax=ax, profile=self.direction, position=self.position, title=title, diff=False, offset=self.offset, vertical_line=self.line, xlim=self.xlim)
            fig.canvas.draw_idle()

        elif event.key == 'enter':
            self.wait = False
            return self.offset

########################### End class DoseAnalysis ############################## 
def line_intersection(line1, line2):
    """ Get the coordinates of the intersection of two lines.

        Parameters
        ----------
        line1 : tuple 
            Coordinates of 2 points defining the first line
            line1 = ((x1,y1),(x2,y2))
        
        line1 : tuple 
            Coordinates of 2 points defining the second line
            line1 = ((x1,y1),(x2,y2)) 
    """
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

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
    """
    Save the dose object to a file using pickle serialization.

    Parameters
    ----------
    dose : object
        The dose object to be saved.
    
    filename : str
        The name of the file to save the dose object to.
    """
    dose.filename = filename
    with open(filename, 'wb') as output:
        pickle.dump(dose, output, pickle.HIGHEST_PROTOCOL)

def load_dose(filename):
    """
    Load a dose object from a file using pickle deserialization.

    Parameters
    ----------
    filename : str
        The name of the file to load the dose object from.

    Returns
    -------
    object
        The loaded dose object.
    """
    with open(filename, 'rb') as input:
        return pickle.load(input)

def load_analysis(filename):
    """
    Load an analysis object from a file, with optional decompression.

    Parameters
    ----------
    filename : str
        The name of the file to load the analysis object from.

    Returns
    -------
    object
        The loaded analysis object.
    """
    print(f"\nLoading analysis file {filename}...")
    try:
        file = bz2.open(filename, 'rb')
        analysis = pickle.load(file)
    except:
        file = open(filename, 'rb')
        analysis = pickle.load(file)
    file.close()
    return analysis

def save_analysis(analysis, filename, use_compression=True):
    """
    Save the analysis object to a file using pickle serialization, with optional compression.

    Parameters
    ----------
    analysis : object
        The analysis object to be saved.
    
    filename : str
        The name of the file to save the analysis object to.
    
    use_compression : bool, optional, default=True
        Whether to compress the file using bz2.
    """
    print(f"\nSaving analysis file as {filename}...")
    if hasattr(analysis, "ruler"): del analysis.ruler
    if use_compression:
        file = bz2.open(filename, 'wb')
    else:
        file = open(filename, 'wb')
    pickle.dump(analysis, file, pickle.HIGHEST_PROTOCOL)
    file.close()

def add_ruler(ax=None):
    """
    Add a ruler to the specified axis for measurement purposes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes, optional
        The axis to add the ruler to. If None, the current axis is used.
        Default is None.

    Returns
    -------
    Ruler
        The created Ruler object.
    """
    if ax is None: ax = plt.gca()
    markerprops = dict(marker='o', markersize=5, markeredgecolor='red')
    lineprops = dict(color='red', linewidth=2)
    ruler = Ruler(ax=ax, useblit=True, markerprops=markerprops, lineprops=lineprops)
    return ruler