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

    from omg_dosimetry.calibration import LUT

    LUT.run_demo()

Usage
-----

This section is used to demonstrate an example for performing gafchromic calibration.

Import :class:`~omg_dosimetry.calibration.LUT` and Path

.. code-block:: python

    from omg_dosimetry.calibration import LUT
    from pathlib import Path
    import matplotlib.pyplot as plt

Define the folder containing the scanned tiff images, and the nominal doses [cGy] imparted to the films

.. note:: Only 16-bit/channel RGB tiff images are supported.

.. code-block:: python

    my_path = Path(r"C:/my/folder")
    doses = [0.0, 100.0, 200.0, 400.0, 650.0, 950.0]

If you don't have an image you can load the demo images:

.. code-block:: python

    from omg_dosimetry import calibration
    my_path = calibration.from_demo_image()

Produce the LUT

.. code-block:: python

    lut = LUT(my_path, doses, crop_top_bottom = 650) # Crop needed because an unwanted border

To display films and ROIs used for calibration

.. code-block:: python

    lut.plot_roi()
    plt.show()

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


Set calibration parameters
--------------------------

For a detalied description see :class:`~omg_dosimetry.calibration.LUT` class.

Daily output factor
===================

Daily output factor could be acounted for when films were exposed. Doses will be corrected as 
doses_corrected = doses * output

.. code-block:: python
 
    from omg_dosimetry import LUT

    lut = LUT(..., output = 1)                           

Lateral correction
==================

Define if lateral scanner response correction is applied.

* **True**: A LUT is computed for every pixel in the scanner lateral direction
* **False**: A single LUT is computed for the scanner.

.. code-block:: python

    from omg_dosimetry import LUT

    lut = LUT(..., lateral_correction = True)

Beam profile correction 
=======================

None to not correct for the shape of the dose profile, or path to a text file containing the shape profile
(position and relative profile value). Data in the first column should be position, given in mm, with 0 being at center.
Second column should be the measured profile relative value [%], normalised to 100 in the center.

.. code-block:: python

    from omg_dosimetry import LUT

    lut = LUT(..., beam_profile = Path(my_path, "BeamProfile.txt"))


Film detection
==============

Define automatic o manual film detection

.. code-block:: python

    lut = LUT(..., film_detect = True)

Crop
====

If film_detect = True: Number of pixels to crop in the top and bottom of the image.
May be required for auto-detection if the glass on the scanner is preventing detection

.. code-block:: python

    lut = LUT(..., crop_top_bottom = 650)

ROI size
========

Define the size of the region of interest over the calibration films.
If film_detect = True: 'auto' to define the size of the ROIs according to the films,
or [width, height] (mm) to define a fixed size.

.. code-block:: python

    lut = LUT(..., roi_size = 'auto')

ROI crop
========

If film_detect = True and roi_size = 'auto': Margin size [mm] to apply on each side
films to define the ROI.

.. code-block:: python

    lut = LUT(..., roi_crop = 3)

Filtering
=========

For image filtering, median filter kernel size to apply on images for noise reduction.

.. code-block:: python

    lut = LUT(..., filt = 3)

Metadata
========

Define general information

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
    
    lut = LUT(..., info = info)

API Documentation
-----------------

LUT class
==========

.. autoclass:: omg_dosimetry.calibration.LUT
    :members: