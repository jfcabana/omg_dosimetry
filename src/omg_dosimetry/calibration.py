# -*- coding: utf-8 -*-
"""
OMG Dosimetry calibration module.

The calibration module computes multichannel calibration curves from scanned films. 

Scanned films are automatically detected and selected, or ROIs can be drawn manually.

The lateral scanner response effect (inhomogeneous response of the scanner along the detector array) can be accounted for by creating separate calibration curves for each pixel along the array.
This requires exposing long film strips and scanning them perpendicular to the scan direction (see demonstration files). 
To account for non-flat beam profiles, a text file containing the relative beam profile shape along the film strips can be given as input to correct for non-uniform dose on the film.
Alternatively, the lateral scanner response correction can be turned off, then a single calibration curve is computed for all pixels. This simpler calibration is adequate if scanning only small films at a reproducible location on the scanner.

Features:

* Automatically loads multiple images in a folder, average multiple copies of same image and stack different scans together.
* Automatically detect films position and size, and define ROIs inside these films.
* Daily output correction
* Beam profile correction
* Lateral scanner response correction
* Save/Load LUT files
* Publish PDF report
    
Written by Jean-Francois Cabana and Luis Alfonso Olivares Jimenez, copyright 2018
Modified by Peter Truong (CISSSO)
Version: 2023-12-06
"""

from pylinac.core.profile import SingleProfile
from pylinac import profile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets  import RectangleSelector
import pickle
import csv
from scipy.signal import medfilt
import os
from pylinac.core import pdf
import io
from scipy.optimize import curve_fit
from random import randint
from scipy.interpolate import UnivariateSpline
from pathlib import Path
import webbrowser
from .imageRGB import load, load_folder, stack_images
import bz2
from .i_o import retrieve_demo_file

class LUT:
    """
    Class for performing gafchromic calibration.
    
    Parameters
    ----------

    path : str
        Path to folder containing scanned tif images of calibration films.
        Multiple scans of the same films should be named (someName)_00x.tif
        These files will be averaged together to increase SNR.

        Files with different basename ('someName1_00x.tif', 'someName2_00x.tif', ...)
        will be stacked side by side. This is to allow scanning films seperately,
        either because they don't fit on the scanner bed all at once, or to have
        the films scanned at the same location to mitigate scanner response inhomogeneities.

    doses : list of floats
        List of nominal doses values that were delivered on the films.

    output : float
        Daily output factor when films were exposed.
        Doses will be corrected as: doses_corr = doses * output

    lateral_correction : boolean
        Define if lateral scanner response correction is applied.
        True: A LUT is computed for every pixel in the scanner lateral direction
        False: A single LUT is computed for the scanner.

        As currently implemented, lateral correction is performed by exposing
        long strips of calibration films with a large uniform field. By scanning
        the strips perpendicular to the scanner direction, a LUT is computed
        for each pixel in the scanner lateral direction. If this method is
        used, it is recommended that beam profile correction be applied also,
        so as to remove the contribution of beam inhomogeneity.

    beam_profile : str
        Full path to beam profile text file that will be used to correct the doses at each pixel position.
        The text file has to be tab seperated containing the position and relative profile value.
        First column should be a position, given in mm, with 0 being at center.

        Second column should be the measured profile relative value [%], normalised to 100 in the center.
        Corrected doses are defined as dose_corr(y) = dose * profile(y),
        where profile(y) is the beam profile, normalized to 100% at beam center
        axis, which is assumed to be aligned with scanner center.

        If set to 'None', the beam profile is assumed to be flat.

    filt : int (must be odd)
        If filt > 0, a median filter of size (filt,filt) is applied to 
        each channel of the scanned image prior to LUT creation.            
        This feature might affect the automatic detection of film strips if
        they are not separated by a large enough gap. In this case, you can
        either use manual ROIs selection, or apply filtering to the LUT during
        the conversion to dose (see tiff2dose module).

    film_detect : boolean
        Define if automatic film position detection is performed.
        True:  The film positions on the image are detected automatically, by finding peaks in the longitudinal and lateral directions.
        False: The user must manually draw the ROIs over the films.

    roi_size : str ('auto') or list of floats ([width, length])
        Define the size of the region of interest over the calibration films.
        Used only when film_detect is set 'auto'.

        'auto': The ROIs are defined automatically by the detected film strips.

        [width, length]: Size (in mm) of the ROIs. The ROIs are set to a fixed
        size at the center of the detected film strips.

    roi_crop : float
        Margins [mm] to apply to the detected film to define the ROIs.
        Used only when both film_detect and roi_size are set to 'auto'.

    crop_top_bottom : float
        Number of pixels to crop in the top and bottom of the image.
        Used only when film_detect is set 'auto'.

        May be required for correct detection of films if a glass plate is placed on top of the films and is preventing detection.

    info : dictionary
        Used to store information about the calibration that will be shown on the calibration report.
        key: value pairs must include "author", "unit", "film_lot", "scanner_id", date_exposed", "date_scanned", "wait_time", "notes"

    Attributes
    ----------
            
    LUT.lut : numpy array
        When lateral correction is applied:

        3D array of size (6, nDoses, nPixel). The first dimension contains 
        [doses, output/profile corrected doses, mean channel, R channel, G channel, B channel]. 
        nDoses is the number of calibration doses used, and
        nPixel is the number of pixels in the lateral scanner direction.
        
        Without lateral correctoin:

        2D array of size (6, nDoses), defined as above, except that a single LUT is stored
        by taking the median values over the ROIs, instead of one LUT for each scanner pixel.
    LUT.channel_mean : 2D array of size (nDoses, nPixel)
        Contains the average RGB value for each dose, at each pixel location.
    LUT.channel_R : 2D array of size (nDoses, nPixel)
        Contains the Red channel value for each dose, at each pixel location.
    LUT.channel_G : 2D array of size (nDoses, nPixel)
        Contains the Gren channel value for each dose, at each pixel location.
    LUT.channel_B : 2D array of size (nDoses, nPixel)
        Contains the Blue channel value for each dose, at each pixel location.
    LUT.doses_corr : 2D array of size (nDoses, nPixel)
        Contains the output and beam profile corrected doses, at each pixel location.
    """

    def __init__(
        self, 
        path=None, 
        doses=None, 
        output=1.0, 
        lateral_correction=False, 
        beam_profile=None,
        filt=3, 
        film_detect=True, 
        roi_size='auto', 
        roi_crop=3.0, 
        info=None, 
        crop_top_bottom=None
        ):
        """Initializer.
        """

        if path is None:
            raise ValueError("You need to provide a path to a folder containing scanned calibration films!")
        if doses is None:
            raise ValueError("You need to provide nominal doses!")
        if info is None:
            info = dict(author='', unit='', film_lot='', scanner_id='', date_exposed='', date_scanned='', wait_time='', notes='')
        
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
        
        # Initialize some things
        self.lut = []
        self.profile = None  
        self.load_images(path, filt)    # load and process images folder
        
        if crop_top_bottom is not None:
            self.img.crop(pixels=crop_top_bottom, edges=('top','bottom'))
            self.img.pad_rgb(pixels=crop_top_bottom, value=1, edges=('top','bottom'))
            
        self.get_longi_profile()        # get the longitudinal profile at the center of scanner
        self.compute_latpos()           # compute absolute position [mm] of pixels in the y direciton, with 0 at scanner center

        if beam_profile is not None:                    # if a beam profile text file is given
            self.profile = get_profile(beam_profile)    # load and store the profile
            
        if film_detect:                 # Detect the films position automatically...
            self.detect_film()                              
        else:                           # or select ROI manually  
            self.select_film()

    @staticmethod
    def run_demo(film_detect = True, show = True) -> None:
        """Run the LUT demo by loading the demo images and print results.
        
        Parameters
        ----------
        film_detect : bool
            True to attempt automatic film detection, or False to make a manual selection.
        show : bools
            Display a summary of the results.
        """

        info = dict(author = 'Demo Physicist',
            unit = 'Demo Linac',
            film_lot = 'XD_1',
            scanner_id = 'Epson 72000XL',
            date_exposed = '2023-01-24 16h',
            date_scanned = '2023-01-25 16h',
            wait_time = '24 hours',
            notes = 'Transmission mode, @300ppp and 16 bits/channel'
           )
        
        # Download demo tif files and save it on demo_files folder.
        retrieve_demo_file("C14_calib-18h-1_001.tif")
        retrieve_demo_file("C14_calib-18h-2_001.tif")

        # Folder containing scanned images
        demo_path = Path(__file__).parent / "demo_files" / "calibration" / "scan"
        outname = 'Demo_calib'                                 ## Name of the calibration file to produce

        #%% Set calibration parameters
        #### Dose
        doses = [0.0, 100.0, 200.0, 400.0, 650.0, 950.0]      ## Nominal doses [cGy] imparted to the films
        output = 1.0                                          ## If necessary, correction for the daily output of the machine

        ### Lateral correction
        lateral_correction = True                             ## True to perform a calibration with lateral correction of the scanner (requires long strips of film)
                                                                # or False for calibration without lateral correction
        beam_profile = retrieve_demo_file("BeamProfile.txt")  ## None to not correct for the shape of the dose profile,
                                                                # or path to a text file containing the shape profile

        ### Film detection
        
        crop_top_bottom = 650   ## If film_detect = True: Number of pixels to crop in the top and bottom of the image.
                                # May be required for auto-detection if the glass on the scanner is preventing detection
        roi_size = 'auto'       ## If film_detect = True: 'auto' to define the size of the ROIs according to the films,
                                # or [width, height] (mm) to define a fixed size.
        roi_crop = 3            ## If film_detect = True and roi_size = 'auto': Margin size [mm] to apply on each side
                                # films to define the ROI.

        ### Image filtering
        filt = 3                ## Median filter kernel size to apply on images for noise reduction

        #%% Produce the LUT
        lut = LUT(path=demo_path, doses=doses, output=output, lateral_correction=lateral_correction, beam_profile=beam_profile,
                                film_detect=film_detect, roi_size=roi_size, roi_crop=roi_crop, filt=filt, info=info, crop_top_bottom = crop_top_bottom)

        #%% View results and save LUT
        #LUT.plot_roi()  # To display films and ROIs used for calibration
        #LUT.plot_fit()  # To display a plot of the calibration curve and the fitted algebraic function
        #lut.publish_pdf(filename=os.path.join(demo_path, outname +'_report.pdf'), open_file=True)            # Generate a PDF report
        save_lut(lut, filename=os.path.join(demo_path.parent, outname + '.pkl'), use_compression=True)  # Save the LUT file. use_compression allows a reduction  
                                                                                                        # in file size by a factor of ~10, but slows down the operation.
        if show:
            lut.show_results(io.BytesIO())

    def load_images(self,path,filt):
        """ Load all images in a folder. Average multiple copies of same image
            together and stack multiple scans side-by-side.
        """
        if os.path.isdir(path):
            images = load_folder(path)                 # load all tiff images in folder and merge multiples copies of same scan        
            img = stack_images(images, axis=1)         # merge different scans side by side   
        elif os.path.isfile(path):
            img = load(path)
        else:
            raise ValueError("ERROR: path is not valid: ", path)

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
        prof = profile.find_peaks(bined)
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
        print("Automatic film detection in progress...")
        xpos = self.longitudinal_profile[0]
        data_longi = self.longitudinal_profile[1]
        n = len(xpos)
        
        # Detect vertical (y) films position and length
        ypos = []
        length = []
        for x in xpos:
            col = np.mean(self.img.array[:,x,:], axis=-1)
            prof = SingleProfile(col)
            prof.invert()
            prof.filter(size=3)
            data = prof.fwxm_data(x=50)
            y = data["center index (rounded)"]
            l = data["width (rounded)"]
            ypos.append(y)
            length.append(l)
            
        # Define ROIs by cropping the film strips...
        if self.roi_size == 'auto':
            width = []
            for i in range(0,n):
                w = data_longi["right_bases"][i] - data_longi["left_bases"][i]
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
        
        self.roi_xpos, self.roi_ypos = [], []
        self.roi_xmin, self.roi_xmax = [], []
        self.roi_ymin, self.roi_ymax = [], []
        self.roi_width, self.roi_length = [], []
        
        plt.figure()
        ax = plt.gca()  
        self.img.plot(ax=ax, show = False)  
        ax.plot((0,self.img.shape[1]),(self.img.center.y,self.img.center.y),'k--')
        ax.set_xlim(0, self.img.shape[1])
        ax.set_ylim(self.img.shape[0],0)
        ax.set_title('Click and drag to draw ROIs manually.\n Press ''c'' to catch the ROI and ''enter'' when finished.')
        print('Click and drag to draw ROIs manually.\n Press ''c'' to catch the ROI and ''enter'' when finished.')


        def print_roi_size(eclick, erelease):
            ax = plt.gca()
            x1, y1 = int(eclick.xdata), int(eclick.ydata)
            x2, y2 = int(erelease.xdata), int(erelease.ydata)

            print(f"ROI size (width, height) mm: ({np.abs(x1-x2)/self.img.dpmm:.1f}, {np.abs(y1-y2)/self.img.dpmm:.1f})")


        def key_c_pressed(event):
            """ Create a ROI when 'c' is pressed."""            
            if event.key in ['c', 'C']:
                ax = plt.gca()
                xmin, xmax, ymin, ymax = map(int, self.rs.extents)
                rect = plt.Rectangle(
                    (xmin,ymin),
                    xmax-xmin,
                    ymax-ymin,
                    fill=True,
                    facecolor='red',
                    edgecolor='black',
                    alpha=0.5)
                ax.add_patch(rect)
                plt.gcf().canvas.draw_idle()

                self.roi_xmin.append(xmin)
                self.roi_xmax.append(xmax)
                self.roi_xpos.append(xmin + int(np.floor((xmax-xmin)/2)))
                self.roi_ymin.append(ymin)
                self.roi_ymax.append(ymax)
                self.roi_ypos.append(ymin + int(np.floor((ymax-ymin)/2)))
                self.roi_width.append(xmax-xmin)
                self.roi_length.append(ymax-ymin)

        self.rs = RectangleSelector(
            ax,
            onselect=print_roi_size,
            useblit=True,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords='pixels',
            interactive=True,
            drag_from_anywhere=True,
            props=dict(facecolor='red', edgecolor='black', alpha=0.4),
            )
        plt.gcf().canvas.mpl_connect('key_press_event', self.press_enter)
        plt.gcf().canvas.mpl_connect('key_press_event', key_c_pressed)
        self.wait = True
        plt.show()  
        while self.wait:    # This while is ejecuted only in interactive mode. press_enter changes self.wait to False 
            plt.pause(5)
        
    def press_enter(self, event):
        """ Continue LUT creation when ''enter'' is pressed. """
        
        if event.key == 'enter':
            plt.close(plt.gcf())
            del self.rs
            self.get_rois()
            self.create_LUT()
            self.wait = False
      
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
        print("Creating LUT...")
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
            # correct doses for beam profile (if given)
            if self.profile is not None:
                for i in range(self.npixel):
                    dose = self.doses_corr[:,i]
                    # interpolate beam profile at each pixel location
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
            # This will be used to compute dose outside of calibrated region, but films should never be placed there.
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
            
    def plot_roi(self, ax=None, show=False):      
        """ Plots the scanned films image overlaid by the ROIs.
        """
        if ax is None:
            plt.figure()
            ax = plt.gca()
        self.img.plot(ax=ax, show=show)  

        for i in range(self.nfilm):
            p = plt.Rectangle( (self.roi_xmin[i],self.roi_ymin[i]), self.roi_width[i], self.roi_length[i], color='r', fill=False ) 
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
                  Defines wether to plot mean curves over all pixels, plot a single curve for each pixel, or both.
                  Only applies when lateral correction is used.
        """
        if ax is None:
            plt.figure()
            ax = plt.gca()
        if self.lateral_correction:
            if mode == 'all' or mode == 'both':
                x = np.mean(self.doses_corr, axis=-1)
                mean = self.channel_mean
                R, G, B = self.channel_R, self.channel_G, self.channel_B
                ax.plot(x,mean,color=(0.6,0.6,0.6),linewidth=1)
                ax.plot(x,R,color=(1,0.6,0.6),linewidth=1)
                ax.plot(x,G,color=(0.6,1,0.6),linewidth=1)
                ax.plot(x,B,color=(0.6,0.6,1),linewidth=1)
            if mode == 'mean' or mode == 'both':
                x = np.mean(self.doses_corr, axis=-1)
                mean = np.mean(self.channel_mean, axis=-1)
                R, G, B = np.mean(self.channel_R, axis=-1), np.mean(self.channel_G, axis=-1), np.mean(self.channel_B, axis=-1)
                ax.plot(x,mean,'k', x,R,'r', x,G,'g', x,B,'b', linewidth=3.0)
        else:
            x = self.doses_corr
            mean = self.channel_mean
            R, G, B = self.channel_R, self.channel_G, self.channel_B
            ax.plot(x,mean,'k', x,R,'r', x,G,'g', x,B,'b', linewidth=3.0)
        ax.set_title('Calibration curves')
        ax.set_xlabel('Dose (cGy)')
        ax.set_ylabel('Normalized pixel value')
            
    def plot_fit(self, ax=None, i=None, show_derivative=False, fit_type='rational', k=3, ext=3, s=0):
        """ Plots the fitted function curve.

            show_derivative : boolean
                In addition to the function curve, the first derivative of the function is displayed.

            fit_type : 'rational' or 'spline'
                Determines the type of function used for fitting.
                'rational' : y = -c + b/(x-a)
                'spline' : Uses the function UnivariateSpline from scipy.interpolate
                'k', 'ext', and 's' are parameters to the UnivariateSpline
        """

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
        
    def show_results(self, savefile = None, show = True):
        """ Display a summary of the results.
        """
        fig = plt.figure(figsize=(8, 8))
        fig.suptitle('Close this figure to continue...', fontsize=10)
        if self.lateral_correction:
            ax1 = plt.subplot2grid((3, 6), (0, 0), colspan=3)
            ax2 = plt.subplot2grid((3, 6), (0, 3), colspan=3)
            ax3 = plt.subplot2grid((3, 6), (1, 0), colspan=6)
            ax4 = plt.subplot2grid((3, 6), (2, 0), colspan=2)
            ax5 = plt.subplot2grid((3, 6), (2, 2), colspan=2)
            ax6 = plt.subplot2grid((3, 6), (2, 4), colspan=2)
            self.plot_roi(ax=ax1)
            self.plot_profile(ax=ax2)
            self.plot_calibration_curves(mode='all',ax=ax3)
            self.plot_fit(i='mean', ax=ax3)
            self.plot_lateral_response(ax=(ax4,ax5,ax6))
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
        if savefile: plt.savefig(savefile)
        if show: plt.show()
 
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
        self.show_results(filename, **kwargs)
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
        title = 'Film Calibration Report'
        canvas = pdf.PylinacCanvas(filename, page_title=title, logo=Path(__file__).parent / 'OMG_Logo.png')
        data = io.BytesIO()
        self.save_analyzed_image(data, show = False)
        canvas.add_image(image_data=data, location=(3, 3.5), dimensions=(15, 15))
        canvas.add_text(text='Film infos:', location=(1, 25.5), font_size=10)
        text = ['Author: {}'.format(self.info['author']),
                'Unit: {}'.format(self.info['unit']),
                'Film lot: {}'.format(self.info['film_lot']),
                'Scanner ID: {}'.format(self.info['scanner_id']),
                'Date exposed: {}'.format(self.info['date_exposed']),
                'Date scanned: {}'.format(self.info['date_scanned']),
                'Wait time: {}'.format(self.info['wait_time']),
               ]
        canvas.add_text(text=text, location=(1, 25), font_size=8)
        canvas.add_text(text='Calibration options:', location=(1, 21.5), font_size=10)
        text = ['Images path: {}'.format(self.path),
                'Nominal doses: ' + np.array2string(self.doses, precision=1, separator=', ', floatmode='fixed', max_line_width=150),
                'Output factor: {}'.format(self.output),
                'Scanner lateral response correction applied: {}'.format(self.lateral_correction),
                'Beam profile correction applied: {}'.format(self.beam_profile),
                'Median filter kernel size: {}'.format(self.filt)
               ]       
        canvas.add_text(text=text, location=(1, 21), font_size=8)
        
        if self.info['notes'] != '':
            canvas.add_text(text='Notes:', location=(1, 2.5), font_size=10)
            canvas.add_text(text=self.info['notes'], location=(1, 2), font_size=8)
        canvas.finish()
        if open_file:
            webbrowser.open(filename)
        
        
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

def get_profile(file):
    """ Load tab seperated txt file containing the position and relative profile value.
        First column should be a position, given in mm, with 0 being at center.
        Second column is the measured profile relative value [%], normalised to 100 in the center.
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
    """ Load a saved LUT file.
    """

    print("Loading LUT file {}...".format(filename))
    try:
        file = bz2.open(filename, 'rb')
        lut = pickle.load(file)
    except:
        file = open(filename, 'rb')
        lut = pickle.load(file)
    file.close()
    return lut

def save_lut(lut, filename, use_compression=True):
    """ Save a LUT to file.

        filename : str
            Complete path to file
        use_compression : boolean
            Whether or not to use bz2 compression to reduce file size
    """
    
    print("Saving LUT file as {}...".format(filename))
    if use_compression:
        file = bz2.open(filename, 'wb')
    else:
        file = open(filename, 'wb')
    pickle.dump(lut, file, pickle.HIGHEST_PROTOCOL)
    file.close()

def from_demo_image() -> Path:
    """Load the demo images and return the path to the content folder."""

    img = retrieve_demo_file("C14_calib-18h-1_001.tif")
    retrieve_demo_file("C14_calib-18h-2_001.tif")
    return img.parent