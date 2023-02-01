# -*- coding: utf-8 -*-
"""
Gafchromic calibration module.

The calibration module computes multichannel calibration curves from scanned films. 
Scanned films are automatically detected and selected, or ROIs can be drawn manually.

The lateral scanner response effect (inhomogeneous response of the scanner along
the detector array) is accounted for by creating separate calibration curves for
each pixel along the array. This requires exposing long film strips and scanning
them perpendicular to the scan direction (see demonstration files). 

To account for non-flat beam profiles, the output from an ICProfiler acquired
at the same time as film exposure can be given as input to correct for beam shape.
Alternatively, the lateral scanner response correction can be turned off, then
a single calibration curve is computed for all pixels.

Features:
    - Automatically loads multiple images in a folder, average multiple copies
      of same image and stack different scans together.
    - Automatically detect film strips position and size, and define ROIs
      inside these film strips.
    - Daily output correction
    - Beam profile correction
    - Lateral scanner response correction
    - Save/Load LUt files
    - Publish PDF report
    
Requirements:
    This module is built as an extension to pylinac package.
    Tested with pylinac 2.0.0 and python 3.5.
    
Written by Jean-Francois Cabana, copyright 2018
"""

from pylinac.core.profile import SingleProfile, MultiProfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets  import RectangleSelector
import imageRGB
import pickle
import csv
from scipy.signal import medfilt
import os
from pylinac.core import pdf
import io
from reportlab.lib.units import cm
from scipy.optimize import curve_fit
from random import randint

from scipy.interpolate import UnivariateSpline

class LUT:
    """ Class for performing gafchromic calibration.
    
    Usage : LUT = calibration.LUT(path='path/to/scanned/tiff/images', doses=[dose1, dose2, ...])
            
    LUT.lut: numpy array
        When lateral correction is applied:
            3D array of size (nDoses, nPixel, 6), where nDoses is the number of calibration doses used,
            nPixel is the number of pixels in the lateral scanner direction, and the last dimension contains
            [doses, output/profile corrected doses, mean channel, R channel, G channel, B channel].
            
        Without lateral correct:
            2D array of size (nDoses, 6), defined as above, except that a single LUT is stored
            by taking the median values over the ROIs, instead of one LUT for each scanner pixel.
            
    LUT.channel_mean: 2D array of size (nDoses, nPixel)
                      Contains the average RGB value for each dose, at each pixel location.
    LUT.channel_R:    2D array of size (nDoses, nPixel)
                      Contains the Red channel value for each dose, at each pixel location.
    LUT.channel_G:    2D array of size (nDoses, nPixel)
                      Contains the Gren channel value for each dose, at each pixel location.
    LUT.channel_B:    2D array of size (nDoses, nPixel)
                      Contains the Blue channel value for each dose, at each pixel location.
    LUT.doses_corr:   2D array of size (nDoses, nPixel)
                      Contains the output and beam profile corrected doses, at each pixel location.

    Attributes
    ----------
    path : str
        Path to folder containing scanned tif images of calibration films.
        Multiple scans of the same films should be named (someName)_00x.tif
        These files will be averaged together to increase SNR.
        Files with different basename ('someName1_00x.tif', 'someName2_00x.tif', ...)
        will be stacked side by side. This is to allow scanning multiple films 
        that can't fit on the scanner bed in one image.
        
    doses : list of floats
        List of nominal doses values that were used to expose films.
        
    output : float
        Daily output factor when films were exposed.
        Doses will be corrected as : doses_corr = doses * output

    lateral_correction : boolean
        Define if lateral scanner response correction is applied.
        True: A LUT is computed for every pixel in the scanner lateral direction
        False: A single LUT is computed for the scanner.
        
        As currently implemented, lateral correction is performed by exposing
        long strips of calibration films to a large uniform field. By scanning
        the strips perpendicular to the scanner direction, a LUT is computed
        for each pixel in the scanner lateral direction. If this method is
        used, it is recommended that beam profile correction be applied also,
        so as to remove the contribution of beam inhomogeneity. As currently
        implemenented, this requires an acqusition using an ICProfiler.
        
    beam_profile : str
        Full path to beam profile file from an ICprofiler, as obtained during
        calibration films exposition. Used to correct the nominal doses at each
        pixel position.
        
        Corrected doses are defined as dose_corr(y) = dose * profile(y),
        where profile(y) is the beam profile, normalized to 1.0 at beam center
        axis, which is assumed to be aligned with scanner center.
        
        If set to 'None', the beam profile is assumed to be flat.
        
    filt : int (must be odd)
        If filt > 0, a median filter of size (filt,filt) is applied to 
        each channel of the scanned image prior to LUT creation.
        
        This feature might affect the automatic detection of film strips if
        they are not separated by a large enough gap. In this case, you can
        either use manual ROIs selection, or apply filting to the LUT during
        the conversion to dose (see tiff2dose module).
        
    film_detect : boolean
        Define if automatic film position detection is performed.
        
        True:  The film positions on the image are detected automatically,
               by finding peaks in the longitudinal and lateral directions.
        False: The user must manually draw the ROIs over the films.
        
    roi_size : str ('auto') or list of floats ([width, length])
        Define the size of the region of interest over the calibration films.
        Used only when film_detect is set 'auto'. 
        
        'auto': The ROIs are defined automatically by the detected film strips.
        [width, length]: Size (in mm) of the ROIs. The ROIs are set to a fixed
                         size at the center of the detectec film strips.
                
    roi_crop : float
        Margins (in mm) to apply to the detected film to define the ROIs.
        Used only when both film_detect and roi_size are set to 'auto'.
    """

    def __init__(self,
                 path=None,
                 doses=None,
                 output=1.0,
                 lateral_correction=False,
                 beam_profile=None,
                 filt=0,
                 film_detect=True,
                 roi_size='auto',
                 roi_crop=2.0,
                 info=None,
                 baseline=None,
                 crop_top_bottom=None
                ):
        
        if path is None:
            raise ValueError("You need to provide a path to a folder containing scanned calibration films!")
        if doses is None:
            raise ValueError("You need to provide nominal doses!")
        if info is None:
            info = dict(author = '', unit = '', film_lot = '', scanner_id = '',
                        date_exposed = '', date_scanned = '', wait_time = '', notes = '')
        
        # Store settings
        self.path = path
        self.doses = np.asarray(doses)
        self.output = output
        self.lateral_correction = lateral_correction
        self.beam_profile = beam_profile
        self.filt = filt
        self.film_detect = film_detect
        self.roi_size = roi_size
        self.roi_crop = roi_crop
        self.info = info
        self.crop_top_bottom = crop_top_bottom
        
        self.baseline = baseline
        if baseline is not None:
            if os.path.isdir(baseline):
                images = imageRGB.load_folder(baseline)     
                img = imageRGB.stack_images(images, axis=1)
            elif os.path.isfile(baseline):
                img = imageRGB.load(baseline)
            self.base = img

        # Initialize some things
        self.lut = []
        self.profile = None  
        self.load_images(path, filt)    # load and process images folder
        
        if crop_top_bottom is not None:
            self.img.crop(pixels=crop_top_bottom, edges=('top','bottom'))
            self.img.pad_rgb(pixels=crop_top_bottom, value=1, edges=('top','bottom'))
            
        self.get_longi_profile()        # get the longitudinal profile at the center of scanner
        self.compute_latpos()           # compute absolute position (mm) of pixels in the y direciton, with 0 at scanner center

        if beam_profile is not None:                    # if an IC Profiler .txt profile file is given
            self.profile = get_profile(beam_profile)    # load the profile
            
        if film_detect:                 # Detect the films position automatically...
            self.detect_film()                              
        else:                           # or select ROI manually  
            self.select_film()
        
    def load_images(self,path,filt):
        """ Load all images in a folder. Average multiple copies of same image
            together and stack multiple scans side-by-side.
        """
        if os.path.isdir(path):
            images = imageRGB.load_folder(path)                 # load all tiff images in folder and merge multiples copies of same scan        
            img = imageRGB.stack_images(images, axis=1)         # merge different scans side by side   
        elif os.path.isfile(path):
            img = imageRGB.load(path)
        else:
            raise ValueError("ERROR: path is not valid: ", path)

#        if self.baseline is not None:
#            img.array = np.log10(img.array / self.base.array)     

        if filt:                                            # apply median filt to each channel if needed
            for i in range (0,3):
                img.array[:,:,i] = medfilt(img.array[:,:,i],  kernel_size=(filt,filt)) 
        self.img = img
        self.npixel = self.img.shape[0]                     # number of pixel in the y direction (lateral scanner direction)
        self.calibrated = np.zeros(self.npixel).astype(int) # array to store the calibration status of each pixel in the y direction
        self.scanned_date = img.date_created()
        
    def get_longi_profile(self, size=20, thresh=0.8):
        """ Detect horizontal (x) films position by taking the profile over a
            small section in the middle of the scanner.
        """
        sect = np.mean(self.img.array[int(self.img.center.y)-size:int(self.img.center.y)+size,:,:], axis=-1)
        row = np.mean(sect, axis=0)
        bined = np.where(row > max(row) * thresh, 0, 1)    # binarize the profile for improved detectability of peaks
        prof = MultiProfile(bined)
        self.longitudinal_profile = prof
        
    def compute_latpos(self):
        """ Defines a correspondance pixel -> position (in mm).
            Center of scanner is defined at y = 0 mm.
        """
        center = self.img.center.y
        pixel = np.asarray(range(self.img.shape[0]))
        y = (pixel - center) / self.img.dpmm
        self.lat_pos = y
        
    def detect_film(self):
        """ Detect the films positions and construct ROIs automatically.
        """
        
        # Find peaks in the profile. You might need to adjust the values below.
        x = self.longitudinal_profile.find_fwxm_peaks(x=50, threshold=0.3, min_distance=0.02)
        xpos = [int(i) for i in x]
        n = len(xpos)
        
        # Detect vertical (y) films position and length
        ypos = []
        length = []
        for x in xpos:
            col = np.mean(self.img.array[:,x,:], axis=-1)
            prof = SingleProfile(col)
            prof.invert()
            prof.filter(size=3)
            y = prof.fwxm_center(x=50)
            l = prof.fwxm(x=50)
            ypos.append(y)
            length.append(l)
            
        # Define ROIs by cropping the film strips...
        if self.roi_size == 'auto':
            longi_profiles = self.longitudinal_profile.subdivide()
            width = []
            for p in longi_profiles:
                w = p.fwxm(x=50)
                width.append(w)
            crop = self.roi_crop * self.img.dpmm
            self.roi_width = (np.floor(np.asarray(width) - crop*2)).astype('int')
            self.roi_length = (np.floor(np.asarray(length) - crop*2)).astype('int')
            width2 = (np.floor(self.roi_width/2)).astype('int')
            length2 = (np.floor(self.roi_length/2)).astype('int')
            
        # or with fixed size at films center
        else:
            width = np.repeat(self.roi_size[0] * self.img.dpmm, n)
            length = np.repeat(self.roi_size[1] * self.img.dpmm, n)
            self.roi_width = width.astype('int')
            self.roi_length = length.astype('int')
            width2 = (np.floor(self.roi_width/2)).astype('int')
            length2 = (np.floor(self.roi_length/2)).astype('int')
        
        self.roi_xpos = np.asarray(xpos).astype('int')
        self.roi_xmin = xpos - width2
        self.roi_xmax = xpos + width2
        self.roi_ypos = np.asarray(ypos).astype('int')
        self.roi_ymin = self.roi_ypos - length2
        self.roi_ymax = self.roi_ypos + length2
        
        # Continue with LUT creation...
        self.get_rois()
        self.create_LUT()   
        
    def select_film(self):
        """ Define ROIs manually by drawing rectangles on the image. """
        
        self.roi_xpos = []
        self.roi_ypos = []
        self.roi_xmin = []
        self.roi_xmax =[]
        self.roi_ymin = []
        self.roi_ymax = []
        self.roi_width = []
        self.roi_length = []
        
        plt.figure()
        ax = plt.gca()  
        self.img.plot(ax=ax)  
        ax.plot((0,self.img.shape[1]),(self.img.center.y,self.img.center.y),'k--')
        ax.set_xlim(0, self.img.shape[1])
        ax.set_ylim(self.img.shape[0],0)
        ax.set_title('Click and drag to draw ROIs manually. Press ''enter'' when finished.')
        print('Click and drag to draw ROIs manually. Press ''enter'' when finished.')
        
        def select_box(eclick, erelease):
            ax = plt.gca()
            x1, y1 = int(eclick.xdata), int(eclick.ydata)
            x2, y2 = int(erelease.xdata), int(erelease.ydata)
            rect = plt.Rectangle( (min(x1,x2),min(y1,y2)), np.abs(x1-x2), np.abs(y1-y2), Fill=False )
            ax.add_patch(rect)
            
            self.roi_xmin.append(min(x1,x2))
            self.roi_xmax.append(max(x1,x2))
            self.roi_xpos.append(min(x1,x2) + int(np.floor(np.abs(x1-x2)/2)))
            self.roi_ymin.append(min(y1,y2))
            self.roi_ymax.append(max(y1,y2))
            self.roi_ypos.append(min(y1,y2) + int(np.floor(np.abs(y1-y2)/2)))
            self.roi_width.append(int(np.abs(x1-x2)))
            self.roi_length.append(int(np.abs(y1-y2)))
        
        self.rs = RectangleSelector(ax, select_box, drawtype='box', useblit=False, button=[1], 
                                    minspanx=5, minspany=5, spancoords='pixels', interactive=True)
        
        plt.gcf().canvas.mpl_connect('key_press_event', self.press_enter)
        
    def press_enter(self, event):
        """ Continue LUT creation when ''enter'' is pressed. """
        
        if event.key == 'enter':
            plt.close(plt.gcf()) 
            del self.rs
            self.get_rois()
            self.create_LUT()    
      
    def get_rois(self):
        """ Get the values and profiles inside ROIs. """
        
        self.nfilm = len(self.roi_xpos)
        self.roi_xpos = np.asarray(self.roi_xpos)
        self.roi_ypos = np.asarray(self.roi_ypos)
        self.roi_xmin = np.asarray(self.roi_xmin)
        self.roi_xmax = np.asarray(self.roi_xmax)
        self.roi_ymin = np.asarray(self.roi_ymin)
        self.roi_ymax = np.asarray(self.roi_ymax)
        self.roi_width = np.asarray(self.roi_width)
        self.roi_length = np.asarray(self.roi_length)
        
        # Vertical limits where calibration data exists for all doses.
        # This determines the calibrated region when lateral response correction is applied.
        # Irrelevant if lateral response correction is not used.
        self.ymin = max(self.roi_ymin)
        self.ymax = min(self.roi_ymax)
        
        # Get ROIs
        self.roi_mean = []
        self.scanner_profile = []
        for i in range(self.nfilm):
            roi = self.img[self.roi_ymin[i]:self.roi_ymax[i], self.roi_xmin[i]:self.roi_xmax[i], :]
            profile = np.median(roi, axis=1)
            median = np.median(profile, axis=0)
            self.roi_mean.append(median)
            
            band = self.img[:, self.roi_xmin[i]:self.roi_xmax[i], :]
            profile = np.median(band, axis=1)
            self.scanner_profile.append(profile)
            
    def create_LUT(self):
        """ Creates the actual LUT array.     
        """
        
        nDose = self.nfilm  
        if nDose != len(self.doses):
            raise ValueError("Number of films does not match number of doses!")
        
        self.doses.sort()                                           # arrange doses in ascending order
        mean_values = np.mean(np.asarray(self.roi_mean),axis=-1)    # get ROIs averaged gray values
        order = np.argsort(-mean_values)                            # arrange mean gray values in descending order (increasing dose)
        self.calibrated[self.ymin:self.ymax] = 1                    # Define region where we have calibration data
        
        # When scanner / beam profile correction is applied, a LUT is stored for each pixel in the y direction
        if self.lateral_correction:
            # correct doses for machine daily output
            self.doses_corr =  np.asarray([self.doses * self.output] * self.npixel).transpose() 
            if self.profile is not None:
                for i in range(self.npixel):
                    dose = self.doses_corr[:,i]
                    # correct doses for beam profile
                    profile = np.interp(self.lat_pos[i], self.profile[:,0], self.profile[:,1]) / 100  
                    dose_corr = dose * profile
                    self.doses_corr[:,i] = dose_corr
            
            # Populate the channels
            arr = np.asarray(self.scanner_profile)    # scanner_profile is the vertical profiles for each ROI over the full scanner heigth
            self.channel_mean = np.mean(arr[order,:,:], axis=-1)
            self.channel_R = arr[order,:,0]
            self.channel_G = arr[order,:,1]
            self.channel_B = arr[order,:,2]
            
            # Replace uncalibrated regions with median values of calibrated pixels
            for i in range(nDose):
                self.channel_mean[i,np.where(self.calibrated == 0)] = np.median(self.channel_mean[i,np.where(self.calibrated == 1)])
                self.channel_R[i,np.where(self.calibrated == 0)] = np.median(self.channel_R[i,np.where(self.calibrated == 1)])
                self.channel_G[i,np.where(self.calibrated == 0)] = np.median(self.channel_G[i,np.where(self.calibrated == 1)])
                self.channel_B[i,np.where(self.calibrated == 0)] = np.median(self.channel_B[i,np.where(self.calibrated == 1)])
            
            lut = np.zeros([6, nDose, self.npixel])
            lut[0,:,:] = np.asarray([self.doses] * self.npixel).transpose()
            lut[1,:,:] = self.doses_corr
            lut[2,:,:] = self.channel_mean
            lut[3,:,:] = self.channel_R
            lut[4,:,:] = self.channel_G
            lut[5,:,:] = self.channel_B
        
        # When no scanner profile correction is applied, we use median values over ROIs
        else:
            self.doses_corr = self.doses * self.output   # Correct doses for machine daily output
            arr = np.asarray(self.roi_mean)
            self.channel_mean = mean_values[order]
            self.channel_R = arr[order,0]
            self.channel_G = arr[order,1]
            self.channel_B = arr[order,2]
            
            lut = np.zeros([6, nDose])
            lut[0,:] = self.doses
            lut[1,:] = self.doses_corr
            lut[2,:] = self.channel_mean
            lut[3,:] = self.channel_R
            lut[4,:] = self.channel_G
            lut[5,:] = self.channel_B
            
        self.lut = lut
    ################### End create_LUT ################
            
    def plot_profile(self,ax=None):
        """ Plots the scanner profile in the x direction, as used for film detection.
        """
        profile = np.mean(self.img.array[int(self.img.center.y)-20:int(self.img.center.y)+20,:,:], axis=0)
        gray = np.mean(profile,axis=-1)
        peaks = np.mean(self.roi_mean, axis=-1)
        
        if ax is None:
            plt.figure()
            ax = plt.gca()      
        ax.plot(profile[:,0],'r')
        ax.plot(profile[:,1],'g')
        ax.plot(profile[:,2],'b')
        ax.plot(gray,'k')
        ax.plot(self.roi_xpos,peaks,'ro')
        ax.set_xlim(0, self.img.shape[1])
        ax.set_ylabel('Channel value')
        ax.set_title('Scanned films peaks profile')
            
    def plot_roi(self, ax=None):      
        """ Plots the scanned films image overlaid by the ROIs.
        """
        if ax is None:
            plt.figure()
            ax = plt.gca()
        self.img.plot(ax=ax)  

        for i in range(self.nfilm):
            p = plt.Rectangle( (self.roi_xmin[i],self.roi_ymin[i]), self.roi_width[i], self.roi_length[i], fill=False, color='r' ) 
            ax.add_patch(p)
            c = plt.Circle((self.roi_xpos[i],self.roi_ypos[i]), 20, color='r', fill=False)
            ax.add_patch(c)
            ax.plot((self.roi_xpos[i],self.roi_xpos[i]),(self.roi_ypos[i]-40,self.roi_ypos[i]+40),'r')
            ax.plot((self.roi_xpos[i]-40,self.roi_xpos[i]+40),(self.roi_ypos[i],self.roi_ypos[i]),'r')
       
        ax.plot((0,self.img.shape[1]),(self.img.center.y,self.img.center.y),'k--')
        ax.plot((0,self.img.shape[1]),(self.ymin,self.ymin),'r--')
        ax.plot((0,self.img.shape[1]),(self.ymax,self.ymax),'r--')
        ax.set_xlim(0, self.img.shape[1])
        ax.set_ylim(self.img.shape[0],0)
        ax.set_title('Scanned films ROIs')
        
    def plot_calibration_curves(self, mode='mean',ax=None):
        """ Plots the LUT calibration curves.
            
            mode: str ('mean', 'all' or 'both')
                  Defines wether to plot mean curves over all pixels, plot 
                  a single curve for each pixel, or both.
        """
        if ax is None:
            plt.figure()
            ax = plt.gca()
        if self.lateral_correction:
            if mode == 'all' or mode == 'both':
                x = np.mean(self.doses_corr, axis=-1)
                mean = self.channel_mean
                R = self.channel_R
                G = self.channel_G
                B = self.channel_B
                ax.plot(x,mean,color=(0.6,0.6,0.6),linewidth=1)
                ax.plot(x,R,color=(1,0.6,0.6),linewidth=1)
                ax.plot(x,G,color=(0.6,1,0.6),linewidth=1)
                ax.plot(x,B,color=(0.6,0.6,1),linewidth=1)
            if mode == 'mean' or mode == 'both':
                x = np.mean(self.doses_corr, axis=-1)
                mean = np.mean(self.channel_mean, axis=-1)
                R = np.mean(self.channel_R, axis=-1)
                G = np.mean(self.channel_G, axis=-1)
                B = np.mean(self.channel_B, axis=-1)
                ax.plot(x,mean,'k', x,R,'r', x,G,'g', x,B,'b', linewidth=3.0)
        else:
            x = self.doses_corr
            mean = self.channel_mean
            R = self.channel_R
            G = self.channel_G
            B = self.channel_B
            ax.plot(x,mean,'k', x,R,'r', x,G,'g', x,B,'b', linewidth=3.0)
        ax.set_title('Calibration curves')
        ax.set_xlabel('Dose (cGy)')
        ax.set_ylabel('Normalized pixel value')
            
    def plot_fit(self, ax=None, i=None, show_derivative=False, fit_type='rational', k=3, ext=3, s=0):
        colors = ['k','r','g','b']
        if i is None:
            i = randint(0,self.npixel)
            
        if ax is None:
            if show_derivative:
                fig, (ax1,ax2) = plt.subplots(1,2)
                ax2.set_title('LUT derivative')
                ax2.set_xlabel('Derivative')
                ax2.set_ylabel('Normalized pixel value')
            else:
                fig, ax1 = plt.subplots(1,1)
            if fit_type == 'rational':
                ax1.set_title('LUT fit (rational), pixel = {}'.format(i))
            elif fit_type == 'spline':
                ax1.set_title('LUT fit (spline), pixel = {}'.format(i))
            ax1.set_xlabel('Dose')
            ax1.set_ylabel('Normalized pixel value')
        else:
            ax1 = ax

        for j in range(2,6):     
            if self.lateral_correction:
                if i == 'mean':
                    p_lut = np.mean(self.lut[:,:,:],axis=-1)
                else:
                    p_lut = self.lut[:,:,i]
                xdata = p_lut[j,:]
                ydata = p_lut[1,:]
            else:
                p_lut = self.lut[:,:]
                xdata = p_lut[j]
                ydata = p_lut[1]
                
            x = np.arange(min(xdata),max(xdata),0.001)   
            if show_derivative:
                if fit_type == 'rational':
                    y, yd = self.get_dose_and_derivative_from_fit(xdata, ydata, x)
                elif fit_type == 'spline':
                    y, yd = self.get_dose_and_derivative_from_spline(xdata, ydata, x, k=k, ext=ext, s=s)
                ax1.plot(ydata,xdata,'o',color=colors[j-2])
                ax1.plot(y,x,color=colors[j-2])  
                ax2.plot(yd,x,color=colors[j-2])  
            else:
                if fit_type == 'rational':
                    y = self.get_dose_from_fit(xdata, ydata, x)
                elif fit_type == 'spline':
                    y = self.get_dose_from_spline(xdata, ydata, x, k=k, ext=ext, s=s)
                ax1.plot(ydata,xdata,'o',color=colors[j-2])
                ax1.plot(y,x,color=colors[j-2])  
        
    def show_results(self, savefile):
        """ Display a summary of the results.
        """
        fig = plt.figure(figsize=(8, 8))
        if self.lateral_correction:
            ax1 = plt.subplot2grid((3, 6), (0, 0), colspan=3)
            ax2 = plt.subplot2grid((3, 6), (0, 3), colspan=3)
            ax3 = plt.subplot2grid((3, 6), (1, 0), colspan=6)
            ax4 = plt.subplot2grid((3, 6), (2, 0), colspan=2)
            ax5 = plt.subplot2grid((3, 6), (2, 2), colspan=2)
            ax6 = plt.subplot2grid((3, 6), (2, 4), colspan=2)
#            ax1 = plt.subplot2grid((5,2), (0, 0), rowspan=2)
#            ax2 = plt.subplot2grid((5,2), (0, 1), rowspan=2)
#            ax3 = plt.subplot2grid((5,2), (2, 0), rowspan=3)
#            ax4 = plt.subplot2grid((5,2), (2, 1))
#            ax5 = plt.subplot2grid((5,2), (3, 1))
#            ax6 = plt.subplot2grid((5,2), (4, 1))
#            ax1 = plt.subplot2grid((4, 6), (0, 0), colspan=3)
#            ax2 = plt.subplot2grid((4, 6), (0, 3), colspan=3)
#            ax3 = plt.subplot2grid((4, 6), (1, 0), colspan=6, rowspan=2)
#            ax4 = plt.subplot2grid((4, 6), (3, 0), colspan=2)
#            ax5 = plt.subplot2grid((4, 6), (3, 2), colspan=2)
#            ax6 = plt.subplot2grid((4, 6), (3, 4), colspan=2)
            self.plot_roi(ax=ax1)
            self.plot_profile(ax=ax2)
            self.plot_calibration_curves(mode='all',ax=ax3)
            self.plot_fit(i='mean', ax=ax3)
            self.plot_lateral_response(ax=(ax4,ax5,ax6))
            fig.tight_layout()
            plt.savefig(savefile)
            plt.show()
        else:
            ax1 = plt.subplot2grid((3, 2), (0, 0))
            ax2 = plt.subplot2grid((3, 2), (0, 1))
            ax3 = plt.subplot2grid((3, 2), (1, 0), rowspan=2, colspan=2)
            self.plot_roi(ax=ax1)
            self.plot_profile(ax=ax2)
            self.plot_fit(ax=ax3)
            ax3.set_title('Calibration curves')
            ax3.set_xlabel('Dose (cGy)')
            ax3.set_ylabel('Normalized pixel value')
            fig.tight_layout()
            plt.savefig(savefile)
            plt.show()
 
    def plot_beam_profile(self, ax=None):
        """ Plot the beam profile.
        """
        if ax is None:
            plt.figure()
            ax = plt.gca()
        ax.set_title('Beam profiles')
        ax.set_xlabel('Lateral scanner position (mm)')
        ax.set_ylabel('Dose')
        x = self.lat_pos
        for i in range(0, self.nfilm):
            y= self.doses_corr[i,:]
            ax.plot(x,y)
            
    def plot_lateral_response(self, ax=None, filt=0):
        """ Plot the raw scanner response for each channels, as a function of lateral position.
        """
        x = self.lat_pos
        if ax is None:
            fig, (ax1,ax2,ax3) = plt.subplots(1,3)
            if filt:
                fig.suptitle('Channel filted lateral response. filt={}'.format(filt))
            else:
                fig.suptitle('Channel raw lateral response.')
        else:
            ax1 = ax[0]
            ax2 = ax[1]
            ax3 = ax[2]
            
        ax1.set_xlabel('Position (mm)')
        ax2.set_xlabel('Position (mm)')
        ax3.set_xlabel('Position (mm)')
        ax1.set_ylabel('Red value')
        ax2.set_ylabel('Green value')
        ax3.set_ylabel('Blue value')

        for i in range(0, self.nfilm):
            if filt:
                y = medfilt(self.channel_R[i,:], kernel_size=filt)
            else:
                y = self.channel_R[i,:]
            ax1.plot(x,y,'r')
            
        for i in range(0,self.nfilm):
            if filt:
                y = medfilt(self.channel_G[i,:], kernel_size=filt)
            else:
                y = self.channel_G[i,:]
            ax2.plot(x,y,'g')
        
        for i in range(0,self.nfilm):
            if filt:
                y = medfilt(self.channel_B[i,:], kernel_size=filt)
            else:
                y = self.channel_B[i,:]
            ax3.plot(x,y,'b')
            
    def save_analyzed_image(self, filename, **kwargs):
        """Save the analyzed image to a file.

        Parameters
        ----------
        filename : str
            The location and filename to save to.
        kwargs
            Keyword arguments are passed to plt.savefig().
        """
        self.show_results(filename)
        fig = plt.gcf()
        fig.savefig(filename)
        plt.close(fig)
            
            
    def publish_pdf(self, filename=None, author=None, unit=None, notes=None, open_file=False):
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
        canvas = pdf.create_pylinac_page_template(filename, analysis_title='Film Calibration Report')
        
        data = io.BytesIO()
        self.save_analyzed_image(data)
        img = pdf.create_stream_image(data)
        canvas.drawImage(img, 3 * cm, 3.5 * cm, width=15 * cm, height=15 * cm, preserveAspectRatio=True)
        
        pdf.draw_text(canvas, x=1 * cm, y=25.5 * cm, text='Film infos:', fontsize=10)
        text = ['Author: {}'.format(self.info['author']),
                'Unit: {}'.format(self.info['unit']),
                'Film lot: {}'.format(self.info['film_lot']),
                'Scanner ID: {}'.format(self.info['scanner_id']),
                'Date exposed: {}'.format(self.info['date_exposed']),
                'Date scanned: {}'.format(self.info['date_scanned']),
                'Wait time: {}'.format(self.info['wait_time']),
               ]
        pdf.draw_text(canvas, x=1 * cm, y=25 * cm, text=text, fontsize=8)

        pdf.draw_text(canvas, x=1 * cm, y=21.5 * cm, text='Calibration options:', fontsize=10)
        text = ['Images path: {}'.format(self.path),
                'Nominal doses: ' + np.array2string(self.doses, precision=1, separator=', ', floatmode='fixed', max_line_width=150),
                'Output factor: {}'.format(self.output),
                'Scanner lateral response correction applied: {}'.format(self.lateral_correction),
                'Beam profile correction applied: {}'.format(self.beam_profile),
                'Median filter kernel size: {}'.format(self.filt)
               ]       
        pdf.draw_text(canvas, x=1 * cm, y=21 * cm, text=text, fontsize=8)
        
        if self.info['notes'] != '':
            pdf.draw_text(canvas, x=1 * cm, y=2.5 * cm, fontsize=10, text="Notes:")
            pdf.draw_text(canvas, x=1 * cm, y=2 * cm, fontsize=8, text=self.info['notes'])
        pdf.finish(canvas, open_file=open_file, filename=filename)
        
        
        ####### Below are the functions used for fitting and interpolating data #######
    def get_dose_from_fit(self, xdata, ydata, x):
        popt, pcov = curve_fit(self.rational_func, xdata, ydata, p0=[0.1, 200, 500], maxfev=1500)
        return self.rational_func(x, *popt)
    
    def get_dose_and_derivative_from_fit(self, xdata, ydata, x):
        popt, pcov = curve_fit(self.rational_func, xdata, ydata, p0=[0.1, 200, 500], maxfev=1500) 
        self.popt = popt
        return self.rational_func(x, *popt), self.drational_func(x, *popt)
    
    def rational_func(self, x, a, b, c):
        return -c + b/(x-a)
    
    def drational_func(self, x, a, b, c):
        return -b/(x-a)**2
    
    def get_dose_from_spline(self, xdata, ydata, x, k=3, ext=3, s=0):
        xdata = xdata[::-1]
        ydata = ydata[::-1]
        f = UnivariateSpline(xdata, ydata, k=k, ext=ext, s=s)
        return f(x)
    
    def get_dose_and_derivative_from_spline(self, xdata, ydata, x, k=3, ext=3, s=0):
        X = np.array(xdata)
        inds = X.argsort()
        xdata = xdata[inds]
        ydata = ydata[inds]
        f = UnivariateSpline(xdata, ydata, k=k, ext=ext, s=s)
        fd = f.derivative()
        return f(x), fd(x)

########################### End class LUT ############################## 


def get_profiler(file):
    """ Load an ICProfler txt file and return the profile over the diagonal
        for the maximum field size. If films were not exposed over the beam
        diagonal, this should be changed here.
    """
    with open(file, 'r') as f:
      reader = csv.reader(f,delimiter='\t')
      content = list(reader)    
    string = ['Detector ID', 'Positive Diagonal Position(cm)', 'Set 1']
    line = content.index(string)
    profile = np.empty((len(content[line+1:]), 2))  
    for i in range(len(content[line+1:])):
        profile[i,0] = float(content[i+line+1][1].replace(',','.')) * 10 # Change from cm to mm  and change sign to correspond to img orientation.
        profile[i,1] = float(content[i+line+1][2].replace(',','.'))      
    return profile

def get_profile(file):
    """ Load tab seperated txt file containing the position and relative profile value.
        First column should be a position, given in mm.
        Second column is the measured profile relative value [%], normalised to 100 in the center.
        Position is relative to scanner center, starting negative from pixel 0.
    """
    with open(file, 'r') as f:
      reader = csv.reader(f,delimiter='\t')
      content = list(reader)    
    profile = np.empty((len(content[:]), 2))  
    for i in range(len(content[:])):
        profile[i,0] = float(content[i][0].replace(',','.'))
        profile[i,1] = float(content[i][1].replace(',','.')) 
    profile = profile[profile[:, 0].argsort()]
    return profile
        
def load_lut_array(filename):
    return np.load(filename)

def save_lut_array(arr, filename):
    np.save(filename, arr)
    
def load_lut(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)

def save_lut(lut, filename):
    with open(filename, 'wb') as output:
        pickle.dump(lut, output, pickle.HIGHEST_PROTOCOL)




         