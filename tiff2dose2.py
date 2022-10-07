# -*- coding: utf-8 -*-
"""
Gafchromic Tiff to Dose module.

The film-to-dose module performs optimized multichannel conversion from scanned gafchromic films to absolute dose.
It uses the optimized multichannel method from Mayer et al (https://doi.org/10.1118/1.3694100)
and calibration curves obtained with the calibration module.

Features:
    - Multiple scans of same film are loaded and averaged automatically
    - Automatic film detection and crop
    - Multichannel optimized conversion to absolute dose (reduced film inhomogeneities/artefacts)
    - Lateral scanner response is accounted for if this feature was turned on during calibration
    - Calibration curves interpolation performed by fitting either a rational function or spline curve
    - Output individual channels dose (R/G/B), as well as optimized dose, mean channel dose and average dose
    - Output metrics for evaluation of dose conversion quality:
     disturbance map, residual error, consistency map
    - Publish PDF report
    
Requirements:
    This module is built as an extension to pylinac package.
    Tested with pylinac 2.0.0, which is compatible with python 3.5.
    
Written by Jean-Francois Cabana, copyright 2018
"""

import os
import calibration
import numpy as np
import imageRGB
from scipy.signal import medfilt
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pickle

from pylinac.core import pdf
import io
from reportlab.lib.units import cm

from matplotlib.widgets  import RectangleSelector

class Gaf:
    """Base class for gafchromic films.
    """

    def __init__(self, path='', lut_file='', img_filt=0, lut_filt=35, fit_type='rational', k=3, ext=3, s=None, info=None, crop_edges=0, clip=None, rot90=0, baseline=None, scale0=False):   
        
        if info is None:
            info = dict(author = '', unit = '', film_lot = '', scanner_id = '',
                        date_exposed = '', date_scanned = '', wait_time = '', notes = '')
        
        # Store settings
        
        self.path = path
        self.lut_file = lut_file
        self.img_filt = img_filt
        self.lut_filt = lut_filt
        self.fit_type = fit_type
        self.info = info
        self.lut = calibration.load_lut(lut_file)
        
#        self.baseline = baseline
#        if self.baseline is not None:
#            if os.path.isdir(baseline):
#                images = imageRGB.load_folder(baseline)     
#                img = imageRGB.stack_images(images, axis=1)
#            elif os.path.isfile(baseline):
#                img = imageRGB.load(baseline)
#            self.base = img
        
        self.load_files(path)
        self.clip = clip

        if rot90:
            self.img.array = np.rot90(self.img.array, k=rot90)
            
        if scale0:
            print('Select ROI of unexposed film')
            self.apply_factor_from_roi()
        
        # Perform the conversion
        self.convert2dose(img_filt=img_filt, lut_filt=lut_filt, fit_type=fit_type, k=k, ext=ext, s=s)
        
        if crop_edges:
            self.dose_ave.crop_edges(threshold=crop_edges)
            self.dose_m.crop_edges(threshold=crop_edges)
            self.dose_r.crop_edges(threshold=crop_edges)
            self.dose_g.crop_edges(threshold=crop_edges)
            self.dose_b.crop_edges(threshold=crop_edges)
            self.dose_opt.crop_edges(threshold=crop_edges)
            self.dose_opt_delta.crop_edges(threshold=crop_edges)
            self.dose_opt_RE.crop_edges(threshold=crop_edges)
            self.dose_rg.crop_edges(threshold=crop_edges)
            self.dose_consistency.crop_edges(threshold=crop_edges)

    def load_files(self, path):
        
        if os.path.isdir(path):
            folder = path
            files = os.listdir(folder)
            filename = os.path.basename(files[0])
            if filename == 'Thumbs.db':
                filename = os.path.basename(files[1])
        else:
            folder = os.path.dirname(path)
            filename = os.path.basename(path)
            files = os.listdir(folder)
            
        file_list = []
        filebase, fileext = os.path.splitext(filename)
        
        if filebase[-3:-1] == '00':
            for file in files:
                name, fileext = os.path.splitext(file)
                if name.lower()[0:-4] == filebase.lower()[0:-4]:
                    file_list.append(os.path.join(folder,name + fileext))
                    
        # If path is a list, we assume they are multiple copies of the same film
        if len(file_list) > 0:
            self.img = imageRGB.load_multiples(file_list)     
        else:
            self.img = imageRGB.load(path)

    def convert2dose(self, img_filt=0, lut_filt=35, fit_type='rational', k=3, ext=3, s=0):        
        """ Performs the conversion to dose.
        """
        
        img = self.img
        lut = self.lut
        ysize = img.shape[0]
        xsize = img.shape[1]
        
        # Check that image and LUT sizes match
        if ysize != lut.npixel:
            raise ValueError("Image dimension does not match LUT resolution!")
        
        # Apply median filter on all image channels if needed
        if img_filt:
            for i in range (0,3):
                img.array[:,:,i] = medfilt(img.array[:,:,i],  kernel_size=(img_filt,img_filt))
        
        # Apply median filter on LUT if needed
        if lut_filt:
            if lut.lateral_correction:
                for i in range (0,len(lut.doses)):  #loop over all doses
                    for j in range (2,6):           #loop over all channels (mean,R,G,B)
                        lut.lut[j,i,:] = medfilt(lut.lut[j,i,:], kernel_size=(lut_filt))
            else:
                pass
        
        # Initialize arrays
        dose_m = np.zeros((ysize, xsize))
        dose_r = np.zeros((ysize, xsize))
        dose_g = np.zeros((ysize, xsize))
        dose_b = np.zeros((ysize, xsize))
        dose_ave = np.zeros((ysize, xsize))
        dose_opt = np.zeros((ysize, xsize))
        delta = np.zeros((ysize, xsize))
        RE = np.zeros((ysize, xsize))
    
        # Convert image to dose one line at a time
        for i in range(0,ysize):
            row = img.array[i,:,:]
            
            if lut.lateral_correction:
                p_lut = lut.lut[:,:,i]
                xdata = p_lut[:,:]
                ydata = p_lut[1,:]
            else:
                p_lut = lut.lut[:,:]
                xdata = p_lut[:]
                ydata = p_lut[1]
                
            if fit_type == 'rational':
                Dm, Am = lut.get_dose_and_derivative_from_fit(xdata[2,:], ydata, np.mean(row, axis=-1))
                Dr, Ar = lut.get_dose_and_derivative_from_fit(xdata[3,:], ydata, row[:,0])
                Dg, Ag = lut.get_dose_and_derivative_from_fit(xdata[4,:], ydata, row[:,1])
                Db, Ab = lut.get_dose_and_derivative_from_fit(xdata[5,:], ydata, row[:,2])
                    
            elif fit_type == 'spline':
                Dm, Am = lut.get_dose_and_derivative_from_spline(xdata[2,:], ydata, np.mean(row, axis=-1), k=k, ext=ext, s=s)
                Dr, Ar = lut.get_dose_and_derivative_from_spline(xdata[3,:], ydata, row[:,0], k=k, ext=ext, s=s)
                Dg, Ag = lut.get_dose_and_derivative_from_spline(xdata[4,:], ydata, row[:,1], k=k, ext=ext, s=s)
                Db, Ab = lut.get_dose_and_derivative_from_spline(xdata[5,:], ydata, row[:,2], k=k, ext=ext, s=s)
                    
            
#            if lut.lateral_correction:
#                p_lut = lut.lut[1:6,:,i]  
#            else:
#                p_lut = lut.lut[1:6,:]  
#                
#            # Get doses  
#            Dm,Dr,Dg,Db,Ar,Ag,Ab = GetDoses(row, p_lut)
            
            # Remove unphysical values
            Dm[Dm < 0] = 0
            Dr[Dr < 0] = 0
            Dg[Dg < 0] = 0
            Db[Db < 0] = 0
            
            Dave = (Dr + Dg + Db) / 3
            
            # Store single channel doses
            dose_m[i,:] = Dm
            dose_r[i,:] = Dr
            dose_g[i,:] = Dg
            dose_b[i,:] = Db
            dose_ave[i,:] = Dave
            
            # Compute terms
            sum_Ak = Ar + Ag + Ab
            sum_Ak2 = Ar**2 + Ag**2 + Ab**2
            sum_DkAk = Dr*Ar + Dg*Ag + Db*Ab
            RS = sum_Ak**2 / sum_Ak2 / 3        # eq. 9
            
            # Compute dose
            dose_temp = ( Dave - RS * sum_DkAk / sum_Ak ) / ( 1 - RS )        # eq. 6
            
            # Remove unphysical values
            dose_temp[dose_temp<0] = 0
            dose_opt[i,:] = dose_temp
            
            # Compute disturbance map
            delta[i,:] = ( (dose_temp-Dr)*Ar + (dose_temp-Dg)*Ag + (dose_temp-Db)*Ab ) / sum_Ak2 # eq. 7
            
            # Compute residual error (eq. 2)
            RE[i,:] = (( Dr + Ar*delta[i,:] - dose_opt[i,:] )**2 +
                      ( Dg + Ag*delta[i,:] - dose_opt[i,:] )**2 + 
                      ( Db + Ab*delta[i,:] - dose_opt[i,:] )**2 )**0.5
             
        
        
        if self.clip is not None:
            dose_m[dose_m > self.clip] = self.clip
            dose_r[dose_r > self.clip] = self.clip
            dose_g[dose_g > self.clip] = self.clip
            dose_b[dose_b > self.clip] = self.clip
            dose_opt[dose_opt > self.clip] = self.clip
            dose_ave[dose_ave > self.clip] = self.clip
        
        
        self.dose_m = imageRGB.load(dose_m, dpi=self.img.dpi) 
        self.dose_r = imageRGB.load(dose_r, dpi=self.img.dpi)   
        self.dose_g = imageRGB.load(dose_g, dpi=self.img.dpi)      
        self.dose_b = imageRGB.load(dose_b, dpi=self.img.dpi)    
        self.dose_ave = imageRGB.load(dose_ave, dpi=self.img.dpi)  
        self.dose_opt = imageRGB.load(dose_opt, dpi=self.img.dpi)   
        self.dose_opt_delta = imageRGB.load(delta, dpi=self.img.dpi)   
        self.dose_opt_RE = imageRGB.load(RE, dpi=self.img.dpi)  
        self.dose_rg = imageRGB.load((dose_r+dose_g)/2., dpi=self.img.dpi)  
        self.dose_consistency = imageRGB.load(((dose_r-dose_g)**2 + (dose_r-dose_b)**2 + (dose_b-dose_g)**2)**0.5, dpi=self.img.dpi)  
        
    def show_results(self):
        max_dose_m = np.percentile(self.dose_m.array,[99.9])[0].round(decimals=-1)
        max_dose_opt = np.percentile(self.dose_opt.array,[99.9])[0].round(decimals=-1)
        clim = [0, max(max_dose_m, max_dose_opt)]   
        fig, ((ax1,ax2,ax3),(ax4,ax5,ax6),(ax7,ax8,ax9)) = plt.subplots(3,3,figsize=(14, 9))

        self.dose_r.plotCB(ax1,clim=clim, title='Red channel dose')
        self.dose_g.plotCB(ax2,clim=clim, title='Green channel dose')
        self.dose_b.plotCB(ax3,clim=clim, title='Blue channel dose')
        self.dose_m.plotCB(ax4,clim=clim, title='Mean channel dose')
        self.dose_rg.plotCB(ax5,clim=clim, title='Red+Green Average dose')
        self.dose_consistency.plotCB(ax6, cmap='gray', title='Consistency')
        self.dose_opt.plotCB(ax7,clim=clim, title='Multichannel optimized dose')
        self.dose_opt_delta.plotCB(ax8, cmap='gray', title='Disturbance')
        self.dose_opt_RE.plotCB(ax9, cmap='gray', title='Residuals')

        fig.tight_layout()
        
    def save_analyzed_image(self, filename, **kwargs):
        """Save the analyzed image to a file.

        Parameters
        ----------
        filename : str
            The location and filename to save to.
        kwargs
            Keyword arguments are passed to plt.savefig().
        """
        self.show_results()
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
        title='Film-to-Dose Report'
        canvas = pdf.create_pylinac_page_template(filename, analysis_title=title)
        
        data = io.BytesIO()
        self.save_analyzed_image(data)
        img = pdf.create_stream_image(data)
        canvas.drawImage(img, 0.5 * cm, 2 * cm, width=20 * cm, height=20 * cm, preserveAspectRatio=True)

        pdf.draw_text(canvas, x=1 * cm, y=25.5 * cm, text='Film infos:', fontsize=12)
        text = ['Author: {}'.format(self.info['author']),
                'Unit: {}'.format(self.info['unit']),
                'Film lot: {}'.format(self.info['film_lot']),
                'Scanner ID: {}'.format(self.info['scanner_id']),
                'Date exposed: {}'.format(self.info['date_exposed']),
                'Date scanned: {}'.format(self.info['date_scanned']),
                'Wait time: {}'.format(self.info['wait_time']),
               ]
        pdf.draw_text(canvas, x=1 * cm, y=25 * cm, text=text, fontsize=10)

        pdf.draw_text(canvas, x=1 * cm, y=21.5 * cm, text='Conversion options:', fontsize=12)
        text = ['Film file: {}'.format(os.path.basename(self.path)),
                'LUT file: {}'.format(os.path.basename(self.lut_file)),
                'Film filter kernel: {}'.format(self.img_filt),
                'LUT filter kernel: {}'.format(self.lut_filt),
                'LUT fit: {}'.format(self.fit_type),
               ]    
        pdf.draw_text(canvas, x=1 * cm, y=21 * cm, text=text, fontsize=10)
        
        if self.info['notes'] != '':
            pdf.draw_text(canvas, x=1 * cm, y=2.5 * cm, fontsize=14, text="Notes:")
            pdf.draw_text(canvas, x=1 * cm, y=2 * cm, text=self.info['notes'])
        pdf.finish(canvas, open_file=open_file, filename=filename)    
        
    def apply_factor_from_roi(self, norm_dose = None):
        """ Define an ROI on an unexposed film to correct for scanner response. """
        
        msg = 'Factor from ROI: Click and drag to draw an ROI manually. Press ''enter'' when finished.'
        self.roi_xmin = []
        self.roi_xmax = []
        self.roi_ymin = []
        self.roi_ymax = []
        
        self.fig = plt.figure()
        ax = plt.gca()  
        self.img.plot(ax=ax)  
        ax.plot((0,self.img.shape[1]),(self.img.center.y,self.img.center.y),'k--')
        ax.set_xlim(0, self.img.shape[1])
        ax.set_ylim(self.img.shape[0],0)
        ax.set_title(msg)
        print(msg)
        
        def select_box(eclick, erelease):
            ax = plt.gca()
            x1, y1 = int(eclick.xdata), int(eclick.ydata)
            x2, y2 = int(erelease.xdata), int(erelease.ydata)
            rect = plt.Rectangle( (min(x1,x2),min(y1,y2)), np.abs(x1-x2), np.abs(y1-y2), Fill=False )
            ax.add_patch(rect)           
            self.roi_xmin = min(x1,x2)
            self.roi_xmax = max(x1,x2)
            self.roi_ymin = min(y1,y2)
            self.roi_ymax = max(y1,y2)
        
        self.rs = RectangleSelector(ax, select_box, drawtype='box', useblit=False, button=[1], 
                                    minspanx=5, minspany=5, spancoords='pixels', interactive=True)    
        self.cid = self.fig.canvas.mpl_connect('key_press_event', self.apply_factor_from_roi_press_enter)
        
        self.wait = True
        while self.wait:
            plt.pause(5)
        return
        
    def apply_factor_from_roi_press_enter(self, event):
        """ Continue when ''enter'' is pressed. """      
        if event.key == 'enter':
            # Get ROIs values
            roi_R = np.median(self.img.array[self.roi_ymin:self.roi_ymax, self.roi_xmin:self.roi_xmax, 0])
            roi_G = np.median(self.img.array[self.roi_ymin:self.roi_ymax, self.roi_xmin:self.roi_xmax, 1])
            roi_B = np.median(self.img.array[self.roi_ymin:self.roi_ymax, self.roi_xmin:self.roi_xmax, 2])
            
            # Unexposed film from calibration
            calib_R = self.lut.channel_R[0]
            calib_G = self.lut.channel_G[0]
            calib_B = self.lut.channel_B[0]
            
            # Compute factors
            self.scale_R = calib_R / roi_R
            self.scale_G = calib_G / roi_G
            self.scale_B = calib_B / roi_B
            
            # Apply scaling
            self.img.array[:,:,0] = self.img.array[:,:,0] * self.scale_R
            self.img.array[:,:,1] = self.img.array[:,:,1] * self.scale_G
            self.img.array[:,:,2] = self.img.array[:,:,2] * self.scale_B
            
            del self.rs                
            self.fig.canvas.mpl_disconnect(self.cid)
            plt.close(self.fig)   
            self.wait = False
            return

def rational_func(x, a, b, c):
    return -c + b/(x-a)

def drational_func(x, a, b, c):
    return -b/(x-a)**2

def GetDoses(row, p_lut):
    row_m = np.mean(row, axis=-1)
    row_r = row[:,0]
    row_g = row[:,1]
    row_b = row[:,2]

    popt_m, pcov_m = curve_fit(rational_func, p_lut[1,:], p_lut[0,:], p0=[0.1, 200, 500], maxfev=1500)
    popt_r, pcov_m = curve_fit(rational_func, p_lut[2,:], p_lut[0,:], p0=[0.1, 200, 500], maxfev=1500)
    popt_g, pcov_m = curve_fit(rational_func, p_lut[3,:], p_lut[0,:], p0=[0.1, 200, 500], maxfev=1500)
    popt_b, pcov_m = curve_fit(rational_func, p_lut[4,:], p_lut[0,:], p0=[0.1, 200, 500], maxfev=1500)
    
    Dm = rational_func(row_m, *popt_m)
    Dr = rational_func(row_r, *popt_r)
    Dg = rational_func(row_g, *popt_g)
    Db = rational_func(row_b, *popt_b)
    
    Ar = drational_func(row_r, *popt_r)
    Ag = drational_func(row_g, *popt_g)
    Ab = drational_func(row_b, *popt_b)
    
    return Dm,Dr,Dg,Db,Ar,Ag,Ab

def save_dose(dose, filename):
    dose.filename = filename
    with open(filename, 'wb') as output:
        pickle.dump(dose, output, pickle.HIGHEST_PROTOCOL)

def load_dose(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)