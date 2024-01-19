===================
Tiff to Dose module
===================

Overview
--------

.. automodule:: omg_dosimetry.tiff2dose
    :no-members:

Running the Demo
----------------

To run the demo, import the main class and run the demo method:

.. plot::
    :include-source:

    from omg_dosimetry.tiff2dose import Gaf

    Gaf.run_demo()

Usage
-----

Import :class:`~omg_dosimetry.tiff2dose.Gaf` and Path:

.. code-block:: python

    from omg_dosimetry.tiff2dose import Gaf
    from pathlib import Path
    import matplotlib.pyplot as plt

Define the folder containing the scanned tiff images, and path to LUT file

.. code-block:: python

    my_path = Path(r"C:/my/folder/QA")
    path_to_lut = Path(r"/my/folder/lut_calib.pkl")

If you don't have an image you can load the demo image and run LUT.run_demo():

.. code-block:: python

    from omg_dosimetry import tiff2dose
    my_path = tiff2dose.from_demo_image()

.. plot::

    from omg_dosimetry.i_o import retrieve_demo_file
    from omg_dosimetry.imageRGB import load
    img_path = retrieve_demo_file("A1A_Multi_6cm_001.tif")
    img = load(img_path)
    img.plot(show=True)

Define the function type used for fitting calibration curve. 'rational' (recommended) or 'spline'

.. code-block:: python

    fit_type = 'rational'

Maximum value [cGy] to limit dose. Useful to avoid very high doses obtained
due to markings on the film.

.. code-block:: python

    clip = 500

Produce the Gaf object

.. code-block:: python

    gaf = Gaf(my_path, lut_file=path_to_lut, fit_type=fit_type, clip=clip)

Display a figure with the different converted dose maps and metrics.

.. code-block:: python

    gaf.show_results(show=True)

API Documentation
=================

Main class
----------

.. autoclass:: omg_dosimetry.tiff2dose.Gaf
    :members:
