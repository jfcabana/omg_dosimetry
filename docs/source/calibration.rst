==================
Calibration module
==================

Overview
--------

.. automodule:: omg_dosimetry.calibration
    :no-members:

Running the Demo
----------------

To run the demo, import the main class and run the demo method:

.. plot::
    :include-source:

    from omg_dosimetry import LUT

    #LUT.run_demo()

Usage
-----

This section is used to demonstrate an example for performing gafchromic calibration.

Import :class:`~omg_dosimetry.calibration.LUT` and Path

.. code-block:: python

    from omg_dosimetry import LUT
    from pathlib import Path
    import matplotlib.pyplot as plt

Define the folder containing the scanned tiff images, and the nominal doses [cGy] imparted to the films

.. code-block:: python

    my_path = Path(r"C:/my/folder")
    doses = [0.0, 100.0, 200.0, 400.0, 650.0, 950.0]

If you don't have an image you can load the demo images:

.. code-block:: python

    #from omg_dosimetry import calibration
    from omg_dosimetry.i_o import retrieve_demo_file
    img_path_1 = retrieve_demo_file("C14_calib-18h-1_001.tif")
    img_path_2 = retrieve_demo_file("C14_calib-18h-2_001.tif")
    my_path = img_path_1.parent
    #my_path = calibration.from_demo_image()

Produce the LUT

.. code-block:: python

    lut = LUT(my_path, doses, crop_top_bottom = 650) # Crop needed because a glass on the scanner

To display films and ROIs used for calibration

.. code-block:: python

    lut.plot_roi(show = True)

.. plot::

    from omg_dosimetry import LUT
    from omg_dosimetry.i_o import retrieve_demo_file
    img_path_1 = retrieve_demo_file("C14_calib-18h-1_001.tif")
    img_path_2 = retrieve_demo_file("C14_calib-18h-2_001.tif")
    my_path = img_path_1.parent
    doses = [0.0, 100.0, 200.0, 400.0, 650.0, 950.0]
    lut = LUT(my_path, doses, crop_top_bottom = 650) # Crop needed because a glass on the scanner
    lut.plot_roi(show = True)

To display a plot of the calibration curve and the fitted algebraic function

.. code-block:: python

    lut.plot_fit()
    plt.show()

.. plot::

    from omg_dosimetry import LUT
    from omg_dosimetry.i_o import retrieve_demo_file
    import matplotlib.pyplot as plt

    img_path_1 = retrieve_demo_file("C14_calib-18h-1_001.tif")
    img_path_2 = retrieve_demo_file("C14_calib-18h-2_001.tif")
    my_path = img_path_1.parent
    doses = [0.0, 100.0, 200.0, 400.0, 650.0, 950.0]
    lut = LUT(my_path, doses, crop_top_bottom = 650) # Crop needed because a glass on the scanner    
    lut.plot_fit()
    plt.show()

Analyze Options
---------------

Define general information and set the folder containing **scanned images**

.. code-block:: python

    info = dict(author = 'Demo Physicist',
                unit = 'Demo Linac',
                film_lot = 'XD_1',
                scanner_id = 'Epson 72000XL',
                date_exposed = '2023-01-24 16h',
                date_scanned = '2023-01-25 16h',
                wait_time = '24 hours',
                notes = 'Transmission mode, @300ppp and 16 bits/channel'
            )
    ## Name of the calibration file to produce
    outname = "Demo_calib"  
    ## Working directory
    my_path = Path(r"C:/my/folder")

Set calibration parameters

.. code-block:: python

    ## Nominal doses [cGy] imparted to the films
    doses = [0.0, 100.0, 200.0, 400.0, 650.0, 950.0]
    ## If necessary, correction for the daily output of the machine  
    output = 1.0                                        



    ### Lateral correction
    lateral_correction = True   ## True to perform a calibration with lateral correction of the scanner (requires long strips of film)
                                ## or False for calibration without lateral correction

    beam_profile = Path(my_path, "BeamProfile.txt")  ## None to not correct for the shape of the dose profile,
                                                         # or path to a text file containing the shape profile

Define automatic o manual film detection

.. code-block:: python

    film_detect = True      ## True to attempt automatic film detection, or False to make a manual selection
    crop_top_bottom = 650   ## If film_detect = True: Number of pixels to crop in the top and bottom of the image.
                            # May be required for auto-detection if the glass on the scanner is preventing detection
    roi_size = 'auto'       ## If film_detect = True: 'auto' to define the size of the ROIs according to the films,
                            # or [width, height] (mm) to define a fixed size.
    roi_crop = 3            ## If film_detect = True and roi_size = 'auto': Margin size [mm] to apply on each side
                            # films to define the ROI.

Image filtering

.. code-block:: python

    filt = 3                ## Median filter kernel size to apply on images for noise reduction

Produce the LUT

.. code-block:: python

    lut = LUT(
        path=path_scan, 
        doses=doses, 
        output=output, 
        lateral_correction=lateral_correction, 
        beam_profile=beam_profile,
        film_detect=film_detect, 
        roi_size=roi_size, 
        roi_crop=roi_crop, 
        filt=filt, 
        info=info, 
        crop_top_bottom = crop_top_bottom
        )


API Documentation
-----------------

Main classes
============

These are the classes a typical user may interface with.

.. autoclass:: omg_dosimetry.calibration.LUT
    :members:
