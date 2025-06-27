# -*- coding: utf-8 -*-
"""
OMG Dosimetry tiff2dose module.

The film-to-dose module performs optimized multichannel conversion from scanned
gafchromic films to absolute dose. It uses the optimized multichannel method from
Mayer et al (https://doi.org/10.1118/1.3694100) and calibration curves obtained
with the calibration module.

Features:
    - Multiple scans of same film are loaded and averaged automatically
    - Automatic film detection and cropping
    - Multichannel optimized conversion to absolute dose (reduced film inhomogeneities/artefacts)
    - Lateral scanner response is accounted for if this feature was turned on during calibration
    - Calibration curves interpolation performed by fitting either a rational function or spline curve
    - Output individual channels dose (R/G/B), as well as optimized dose, mean channel dose and average dose
    - Output metrics for evaluation of dose conversion quality: disturbance map, residual error, consistency map
    - Publish PDF report

Written by Jean-Francois Cabana, copyright 2018.

Modified by Peter Truong (CISSSO) and Luis Alfonso Olivares Jimenez.

Version: 2025-06-27
"""

import os
import numpy as np
from scipy.signal import medfilt
import matplotlib.pyplot as plt
import pickle
from pylinac.core import pdf
import io
from matplotlib.widgets import RectangleSelector, MultiCursor
import webbrowser
from pathlib import Path
from .imageRGB import load, load_multiples
from .calibration import load_lut, LUT
from .i_o import retrieve_demo_file


class Gaf:
    """
    Base class for gafchromic films converted to dose.

    Parameters
    ----------

    path : str
        File path of scanned tif images of film to convert to dose.
        Multiple scans of the same films should be named (someName)_00x.tif
        These files will be averaged together to increase SNR.

    lut_file : str
        File path to LUT film to use for dose conversion.

    img_filt : int, optional
        Kernel size of median filter to apply to image before conversion to dose.
        Default is 0.

    lut_filt : int, optional
        Kernel size of median filter to apply to LUT data before conversion to dose.
        Used only on LUT with lateral scanner correction applied.
        Default is 35.

    fit_type : 'rational' or 'spline'
        Function type used for fitting calibration curve.
        Default is 'rational'.
        If fit_type = "spline", the fitted function is UnivariateSpline from
        scipy.interpolate.
        'k', 'ext', and 's' are parameters to pass to this function.

    k : int, optional
        Degree of the smoothing spline.  Must be 1 <= `k` <= 5.
        ``k = 3`` is a cubic spline. Default is 3.

    s : float or None, optional
        Positive smoothing factor used to choose the number of knots.  Number
        of knots will be increased until the smoothing condition is satisfied::
        sum((w[i] * (y[i]-spl(x[i])))**2, axis=0) <= s
        If `s` is None, ``s = len(w)`` which should be a good value if
        ``1/w[i]`` is an estimate of the standard deviation of ``y[i]``.
        If 0, spline will interpolate through all data points. Default is None.

    ext : int or str, optional
        Controls the extrapolation mode for elements not in the interval defined by
        the knot sequence.
        * if ext=0 or 'extrapolate', return the extrapolated value.
        * if ext=1 or 'zeros', return 0
        * if ext=2 or 'raise', raise a ValueError
        * if ext=3 of 'const', return the boundary value.
        Default is 0.

    info : dictionary, optional
        Used to store information that will be shown on the PDF report.
        key:value pairs must include "author", "unit", "film_lot", "scanner_id",
        date_exposed", "date_scanned", "wait_time", "notes"
        Default is None.

    crop_edges : int, optional
        If not 0, dose arrays will be cropped to remove empty areas around the film.
        The actual value determines the threshold used for detecting what is
        considered non-film area. Default is 0.

    clip : float, optional
        Maximum value [cGy] to limit dose.
        Useful to avoid very high doses obtained due to markings on the film.
        Default is None.
        
    flipLR : bool, optional, default=False
        Whether or not to flip the film dose horizontally to match reference dose orientation.

    flipUD : bool, optional, default=False
        Whether or not to flip the film dose vertically to match reference dose orientation.

    rot90 : int, optional
        Number of 90 degrees rotations to apply to the image.
        Default is 0.

    Attributes
    ----------

    gaf.dose_r:             ArrayImage object of dose from red channnel
    gaf.dose_g:             ArrayImage object of dose from green channnel
    gaf.dose_b:             ArrayImage object of dose from blue channnel
    gaf.dose_m:             ArrayImage object of dose from mean channnel (dose_m = dose_((r+g+b/3)))
    gaf.dose_ave:           ArrayImage object of averaged dose from channels R+G+B ((dose_ave = dose_r + dose_g + dose_b) / 3)
    gaf.dose_rg:            ArrayImage object of averaged dose from channels R+G ((dose_ave = dose_r + dose_g) / 2)
    gaf.dose_opt:           ArrayImage object of optimized dose (eq. 6 in Mayer et al)
    gaf.dose_opt_delta:     ArrayImage object of disturbance map (eq. 7 in Mayer et al)
    gaf.dose_opt_RE:        ArrayImage object of residual error (eq. 2 in Mayer et al)
    gaf.dose_consistency:   ArrayImage object of disagreement between individual channels (RMSE of (dose_r-dose_g) + (dose_r-dose_b) + (dose_b-dose_g))

    Example
    -------

    .. code-block:: python

        gaf = tiff2dose.Gaf(
            path='path/to/scanned/tiff/images',
            lut_file='path/to/calibration/file'
        )

    """

    def __init__(
        self,
        path='',
        lut_file='',
        img_filt=0,
        lut_filt=35,
        fit_type='rational',
        k=3,
        ext=3,
        s=None,
        info=None,
        crop_edges=0,
        clip=None,
        flipLR=False,
        flipUD=False,
        rot90=0
    ):

        if info is None:
            info = dict(
                author='',
                unit='',
                film_lot='',
                scanner_id='',
                date_exposed='',
                date_scanned='',
                wait_time='',
                notes=''
            )
        self.path = path
        self.lut_file = lut_file
        self.img_filt = img_filt
        self.lut_filt = lut_filt
        self.fit_type = fit_type
        self.info = info
        self.lut = load_lut(lut_file)
        self.load_files(path)
        self.clip = clip
        if flipLR: self.img.array = np.fliplr(self.img.array)
        if flipUD: self.img.array = np.flipud(self.img.array)
        if rot90: self.img.array = np.rot90(self.img.array, k=rot90)

        self.convert2dose(
            img_filt=img_filt,
            lut_filt=lut_filt,
            fit_type=fit_type,
            k=k,
            ext=ext,
            s=s
        )

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

    @staticmethod
    def run_demo(show = True) -> None:
        """Run the Gaf demo by loading the demo image and print results.

        Parameters
        ----------
        show : bools
            Display a summary of the results.
        """

        # Define general information
        info = dict(
            author='Demo Physicist',
            unit='Demo Linac',
            film_lot='XD_1',
            scanner_id='Epson 72000XL',
            date_exposed='2023-01-24 16h',
            date_scanned='2023-01-25 16h',
            wait_time='24 hours',
            notes='Transmission mode, @300ppp and 16 bits/channel'
        )

        # Download demo tif files and save it on demo_files folder.
        retrieve_demo_file("A1A_Multi_6cm_001.tif")

        # Folder containing scanned image
        demo_path = Path(__file__).parent / "demo_files" / "tiff2dose" / "scan"

        # Name of the output file to produce
        # outname = "Demo_dose"

        # Path to LUT film to use
        lut_file = from_demo_lut()

        # Function type used for fitting calibration curve. 'rational' (recommended)
        # or 'spline'
        fit_type = 'rational'

        # Maximum value [cGy] to limit dose. Useful to avoid very high doses obtained
        # due to markings on the film.
        clip = 500

        # Perform dose conversion
        gaf1 = Gaf(
            path=demo_path,
            lut_file=lut_file,
            fit_type=fit_type,
            info=info,
            clip=clip
        )

        # Save dose and PDF report
        # filename_tif = demo_path.parent / str(outname + ".tif")

        # We save the optimized dose (dose_opt). Other options include
        # individual channels (dose_r, dose_g, dose_b) and individual
        # channels doses average (dose_ave). gaf1.dose_opt.save(filename_tif)

        # filename_pdf = demo_path / str(outname + ".pdf")
        # gaf1.publish_pdf(str(filename_pdf), open_file=True)

        if show:
            gaf1.show_results(io.BytesIO())

    def load_files(self, path):
        """
        Load image files found in path.
        If path is a directory, it is assumed to contains multiples scans of a
        single film. If a directory contains scans of multiple films, then path
        should be a full path to a single image. Files sharing the same filename
        but ending with _00x in this directory are assumed to be scans of the
        same film and will be averaged together to increse SNR.
        """

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

        self.filename = filename
        file_list = []
        filebase, fileext = os.path.splitext(filename)
        if filebase[-3: -1] == '00':
            for file in files:
                name, fileext = os.path.splitext(file)
                if name.lower()[0: -4] == filebase.lower()[0: -4]:
                    file_list.append(os.path.join(folder, name + fileext))

        # If path is a list, we assume they are multiple copies of the same film
        if len(file_list) > 0:
            self.img = load_multiples(file_list)
        else:
            self.img = load(path)

    def convert2dose(
        self,
        img_filt=0,
        lut_filt=35,
        fit_type='rational',
        k=3,
        ext=3,
        s=0
    ):
        """
        Performs the conversion of scanned image to dose [cGy].
        """

        img = self.img
        lut = self.lut
        ysize = img.shape[0]
        xsize = img.shape[1]

        # Check that image and LUT sizes match (if lateral correction is used)
        if lut.lateral_correction:
            if ysize != lut.npixel:
                raise ValueError("Image dimension does not match LUT resolution!")

        # Apply median filter on all image channels if needed
        if img_filt:
            for i in range(0, 3):
                img.array[:, :, i] = medfilt(img.array[:, :, i],kernel_size=(img_filt, img_filt))

        # Apply median filter on LUT if needed
        if lut_filt:
            if lut.lateral_correction:
                for i in range(0, len(lut.doses)):  # loop over all doses
                    for j in range(2, 6):  # loop over all channels (mean, R, G, B)
                        lut.lut[j, i, :] = medfilt(lut.lut[j, i, :], kernel_size=(lut_filt))
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
        for i in range(0, ysize):
            row = img.array[i, :, :]
            if lut.lateral_correction:
                p_lut = lut.lut[:, :, i]
                xdata = p_lut[:, :]
                ydata = p_lut[1, :]

            else:
                p_lut = lut.lut[:, :]
                xdata = p_lut[:]
                ydata = p_lut[1]

            if fit_type == 'rational':
                Dm, Am = lut.get_dose_and_derivative_from_fit(xdata[2, :],ydata,np.mean(row, axis=-1))
                Dr, Ar = lut.get_dose_and_derivative_from_fit(xdata[3, :], ydata, row[:, 0])
                Dg, Ag = lut.get_dose_and_derivative_from_fit(xdata[4, :], ydata, row[:, 1])
                Db, Ab = lut.get_dose_and_derivative_from_fit(xdata[5, :], ydata, row[:, 2])

            elif fit_type == 'spline':
                Dm, Am = lut.get_dose_and_derivative_from_spline(xdata[2, :], ydata, np.mean(row, axis=-1), k=k, ext=ext, s=s)
                Dr, Ar = lut.get_dose_and_derivative_from_spline(xdata[3, :], ydata, row[:, 0], k=k, ext=ext, s=s)
                Dg, Ag = lut.get_dose_and_derivative_from_spline(xdata[4, :], ydata, row[:, 1], k=k, ext=ext, s=s)
                Db, Ab = lut.get_dose_and_derivative_from_spline(xdata[5, :], ydata, row[:, 2], k=k, ext=ext, s=s)

            # Remove unphysical values
            Dm[Dm < 0], Dr[Dr < 0], Dg[Dg < 0], Db[Db < 0] = 0, 0, 0, 0

            # Store single channel doses
            Dave = (Dr + Dg + Db) / 3
            dose_m[i, :] = Dm
            dose_r[i, :] = Dr
            dose_g[i, :] = Dg
            dose_b[i, :] = Db
            dose_ave[i, :] = Dave

            # Compute terms
            sum_Ak = Ar + Ag + Ab
            sum_Ak2 = Ar**2 + Ag**2 + Ab**2
            sum_DkAk = Dr*Ar + Dg*Ag + Db*Ab
            RS = sum_Ak**2 / sum_Ak2 / 3        # eq. 9

            # Compute dose
            dose_temp = ( Dave - RS * sum_DkAk / sum_Ak ) / ( 1 - RS )        # eq. 6

            # Remove unphysical values
            dose_temp[dose_temp<0] = 0
            dose_opt[i, :] = dose_temp

            # Compute disturbance map
            delta[i, :] = ( (dose_temp-Dr)*Ar + (dose_temp-Dg)*Ag + (dose_temp-Db)*Ab ) / sum_Ak2  # eq. 7

            # Compute residual error (eq. 2)
            RE[i, :] = (( Dr + Ar*delta[i, :] - dose_opt[i, :] )**2 +
                        ( Dg + Ag*delta[i, :] - dose_opt[i, :] )**2 + 
                        ( Db + Ab*delta[i, :] - dose_opt[i, :] )**2 )**0.5

        if self.clip is not None:
            dose_m[dose_m > self.clip] = self.clip
            dose_r[dose_r > self.clip] = self.clip
            dose_g[dose_g > self.clip] = self.clip
            dose_b[dose_b > self.clip] = self.clip
            dose_opt[dose_opt > self.clip] = self.clip
            dose_ave[dose_ave > self.clip] = self.clip

        self.dose_m = load(dose_m, dpi=self.img.dpi)
        self.dose_r = load(dose_r, dpi=self.img.dpi)
        self.dose_g = load(dose_g, dpi=self.img.dpi)
        self.dose_b = load(dose_b, dpi=self.img.dpi)
        self.dose_ave = load(dose_ave, dpi=self.img.dpi)
        self.dose_opt = load(dose_opt, dpi=self.img.dpi)
        self.dose_opt_delta = load(delta, dpi=self.img.dpi)
        self.dose_opt_RE = load(RE, dpi=self.img.dpi)
        self.dose_rg = load((dose_r + dose_g)/2., dpi=self.img.dpi)
        self.dose_consistency = load(
            ((dose_r - dose_g)**2 + (dose_r - dose_b)**2 + (dose_b - dose_g)**2)**0.5,
            dpi=self.img.dpi
        )

    def show_results(self, savefile = None, show = True):
        """
        Display a figure with the different converted dose maps and metrics.
        """

        max_dose_m = np.percentile(self.dose_m.array, [99.9])[0].round(decimals=-1)
        max_dose_opt = np.percentile(self.dose_opt.array, [99.9])[0].round(decimals=-1)
        clim = [0, max(max_dose_m, max_dose_opt)]
        fig, ((ax1,ax2,ax3), (ax4,ax5,ax6), (ax7,ax8,ax9)) = plt.subplots(3, 3, figsize=(14, 9), sharex=True, sharey=True)
        axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]

        self.dose_r.plot(ax1, clim=clim, title='Red channel dose', colorbar=True)
        self.dose_g.plot(ax2, clim=clim, title='Green channel dose', colorbar=True)
        self.dose_b.plot(ax3, clim=clim, title='Blue channel dose', colorbar=True)
        self.dose_m.plot(ax4, clim=clim, title='Mean channel dose', colorbar=True)
        self.dose_rg.plot(ax5, clim=clim, title='Red+Green Average dose', colorbar=True)

        self.dose_consistency.plot(ax6,
            clim = [0, np.percentile(self.dose_consistency.array, [99.5])[0]],
            cmap='gray',
            title='Consistency', 
            colorbar=True
        )

        self.dose_opt.plot(
            ax7,
            clim=clim,
            title='Multichannel optimized dose',
            colorbar=True
        )

        self.dose_opt_delta.plot(
            ax8,
            clim = [np.percentile(self.dose_opt_delta.array, [0.5])[0],
                    np.percentile(self.dose_opt_delta.array, [99.5])[0]],
            cmap='gray',
            title='Disturbance',
            colorbar=True
        )

        self.dose_opt_RE.plot(
            ax9,
            clim = [0, np.percentile(self.dose_opt_RE.array, [99.5])[0]],
            cmap='gray',
            title='Residuals',
            colorbar=True
        )

        fig.tight_layout()

        if savefile:
            plt.savefig(savefile)
        if show:
            plt.multi = MultiCursor(None, (axes), color='r', lw=1, horizOn=True)
            plt.show()

    def save_analyzed_image(self, filename, **kwargs):
        """
        Save the analyzed image to a file.

        Parameters
        ----------
        filename : str
            The location and filename to save to.
        kwargs
            Keyword arguments are passed to plt.savefig().
        """
        self.show_results(**kwargs)
        fig = plt.gcf()
        fig.savefig(filename)
        plt.close(fig)

    def publish_pdf(self, filename=None, author=None, unit=None, notes=None, open_file=False):
        """
        Publish a PDF report of the calibration. The report includes basic
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
        open_file : bool, optional
            Whether or not to open the PDF file after it is created.
            Default is False.
        """
        if filename is None:
            filename = os.path.join(self.path, 'Report.pdf')
        title='Film-to-Dose Report'
        # canvas = pdf.create_pylinac_page_template(filename, analysis_title=title)
        canvas = pdf.PylinacCanvas(
            filename,
            page_title=title,
            logo=Path(__file__).parent / 'OMG_Logo.png'
        )

        data = io.BytesIO()
        self.save_analyzed_image(data, show = False)
        canvas.add_image(image_data=data, location=(0.5, 2), dimensions=(20, 20))
        canvas.add_text(text='Film infos:', location=(1, 25.5), font_size=12)
        text = [
            'Author: {}'.format(self.info['author']),
            'Unit: {}'.format(self.info['unit']),
            'Film lot: {}'.format(self.info['film_lot']),
            'Scanner ID: {}'.format(self.info['scanner_id']),
            'Date exposed: {}'.format(self.info['date_exposed']),
            'Date scanned: {}'.format(self.info['date_scanned']),
            'Wait time: {}'.format(self.info['wait_time']),
        ]
        canvas.add_text(text=text, location=(1, 25), font_size=10)
        canvas.add_text(text='Conversion options:', location=(1, 21.5), font_size=12)
        text = [
            'Film file: {}'.format(os.path.basename(self.filename)),
            'LUT file: {}'.format(os.path.basename(self.lut_file)),
            'Film filter kernel: {}'.format(self.img_filt),
            'LUT filter kernel: {}'.format(self.lut_filt),
            'LUT fit: {}'.format(self.fit_type),
        ]
        canvas.add_text(text=text, location=(1, 21), font_size=10)

        if self.info['notes'] != '':
            canvas.add_text(text='Notes:', location=(1, 2.5), font_size=14)
            canvas.add_text(text=self.info['notes'], location=(1, 2), font_size=14)
        canvas.finish()
        if open_file:
            webbrowser.open(filename)

    # def apply_factor_from_roi(self, norm_dose = None):
    #     """ Define an ROI on an unexposed film to correct for scanner response. """

    #     msg = 'Factor from ROI: Click and drag to draw an ROI manually. Press ''enter'' when finished.'
    #     self.roi_xmin = []
    #     self.roi_xmax = []
    #     self.roi_ymin = []
    #     self.roi_ymax = []

    #     self.fig = plt.figure()
    #     ax = plt.gca()
    #     self.img.plot(ax=ax)
    #     ax.plot((0,self.img.shape[1]),(self.img.center.y,self.img.center.y),'k--')
    #     ax.set_xlim(0, self.img.shape[1])
    #     ax.set_ylim(self.img.shape[0],0)
    #     ax.set_title(msg)
    #     print(msg)

    #     def select_box(eclick, erelease):
    #         ax = plt.gca()
    #         x1, y1 = int(eclick.xdata), int(eclick.ydata)
    #         x2, y2 = int(erelease.xdata), int(erelease.ydata)
    #         rect = plt.Rectangle( (min(x1,x2),min(y1,y2)), np.abs(x1-x2), np.abs(y1-y2), Fill=False )
    #         ax.add_patch(rect)
    #         self.roi_xmin = min(x1,x2)
    #         self.roi_xmax = max(x1,x2)
    #         self.roi_ymin = min(y1,y2)
    #         self.roi_ymax = max(y1,y2)

    #     self.rs = RectangleSelector(ax, select_box, drawtype='box', useblit=False, button=[1],
    #                                 minspanx=5, minspany=5, spancoords='pixels', interactive=True)
    #     self.cid = self.fig.canvas.mpl_connect('key_press_event', self.apply_factor_from_roi_press_enter)

    #     self.wait = True
    #     while self.wait: plt.pause(5)
    #     return

    # def apply_factor_from_roi_press_enter(self, event):
    #     """ Continue when ''enter'' is pressed. """
    #     if event.key == 'enter':
    #         # Get ROIs values
    #         roi_R = np.median(self.img.array[self.roi_ymin:self.roi_ymax, self.roi_xmin:self.roi_xmax, 0])
    #         roi_G = np.median(self.img.array[self.roi_ymin:self.roi_ymax, self.roi_xmin:self.roi_xmax, 1])
    #         roi_B = np.median(self.img.array[self.roi_ymin:self.roi_ymax, self.roi_xmin:self.roi_xmax, 2])

    #         # Unexposed film from calibration
    #         calib_R = self.lut.channel_R[0]
    #         calib_G = self.lut.channel_G[0]
    #         calib_B = self.lut.channel_B[0]

    #         # Compute factors
    #         self.scale_R = calib_R / roi_R
    #         self.scale_G = calib_G / roi_G
    #         self.scale_B = calib_B / roi_B

    #         # Apply scaling
    #         self.img.array[:,:,0] = self.img.array[:,:,0] * self.scale_R
    #         self.img.array[:,:,1] = self.img.array[:,:,1] * self.scale_G
    #         self.img.array[:,:,2] = self.img.array[:,:,2] * self.scale_B

    #         del self.rs
    #         self.fig.canvas.mpl_disconnect(self.cid)
    #         plt.close(self.fig)
    #         self.wait = False
    #         return

    def normalize_dose_opt(self, norm_dose, norm_film_path = None):
        self.norm_film_dose = load(norm_film_path) if norm_film_path else None
        self.apply_factor_from_roi(norm_dose = norm_dose)       # Function adjusted to solely normalize self.dose_opt

    def cleanup(self):
        if hasattr(self, "rs"): del self.rs    
        if hasattr(self, "cursor"): del self.cursor    
        if self.fig:
            self.fig.canvas.mpl_disconnect(self.cid)
            plt.close(self.fig)

    def apply_factor_from_roi(self, norm_dose):
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
        (self.norm_film_dose if self.norm_film_dose else self.dose_opt).plot(ax=ax)
        ax.set_title(msg)
        print(msg)
        
        def select_box(eclick, erelease):
            x1, y1 = int(eclick.xdata), int(eclick.ydata)
            x2, y2 = int(erelease.xdata), int(erelease.ydata)
            self.roi_xmin, self.roi_xmax = min(x1,x2), max(x1,x2)
            self.roi_ymin, self.roi_ymax = min(y1,y2), max(y1,y2)
        
        self.rs = RectangleSelector(ax, select_box, useblit=True, button=[1], minspanx=5, minspany=5, spancoords='pixels', interactive=True)  
        self.cid = self.fig.canvas.mpl_connect('key_press_event', self.apply_factor_from_roi_press_enter)
        
        self.wait = True
        while self.wait: plt.pause(1)
        self.cleanup()

    def apply_factor_from_roi_press_enter(self, event):
        """ Function called from apply_factor_from_roi() when ''enter'' is pressed. """      
        if event.key == 'enter':
            film_dose_array = self.norm_film_dose.array if self.norm_film_dose else self.dose_opt.array
            roi_film = np.median(film_dose_array[self.roi_ymin:self.roi_ymax, self.roi_xmin:self.roi_xmax])
  
            factor = self.norm_dose / roi_film       
            self.apply_film_factor(film_dose_factor = factor)
            self.wait = False
            
    def apply_film_factor(self, film_dose_factor = None):
        """ Apply a normalisation factor to film dose. """
        if film_dose_factor:
            self.film_dose_factor = film_dose_factor
            self.dose_opt.array *= film_dose_factor
            print(f"\nApplied film normalisation factor = {film_dose_factor:.2f}")

def rational_func(x, a, b, c):
    return -c + b/(x-a)


def drational_func(x, a, b, c):
    return -b/(x-a)**2


def save_dose(dose, filename):
    dose.filename = filename
    with open(filename, 'wb') as output:
        pickle.dump(dose, output, pickle.HIGHEST_PROTOCOL)


def load_dose(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)


def from_demo_image() -> Path:
    """Load the demo images and return the path to the content folder."""

    img = retrieve_demo_file("A1A_Multi_6cm_001.tif")
    return img.parent


def from_demo_lut() -> Path:
    """Load the demo lut and return the path to the file."""

    lut_file_path = Path(__file__).parent / "demo_files" / "calibration" / "Demo_calib.pkl"
    if not lut_file_path.exists():
        print("Running LUT.run_demo()...")
        LUT.run_demo(show=False)

    return lut_file_path
