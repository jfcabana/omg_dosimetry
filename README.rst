OMG Dosimetry
=============

.. image:: https://github.com/jfcabana/omg_dosimetry/blob/OMG_master/src/omg_dosimetry/OMG_Logo.png?raw=true
    :width: 100%
    :target: https://github.com/jfcabana/omg_dosimetry
    :align: center

With OMG Dosimetry, you can easily perform film calibration, film-to-dose conversion, and dose analysis.

Demonstration files are provided for each module to get you started quickly. 
Code is heavily commented so you can follow along and adapt it for your personal usage.

Documentation
-------------

To get started, install the package, view the API docs, visit the `Full Documentation <https://omg-dosimetry.readthedocs.io/en/latest/>`_ on Read The Docs.

Gafchromic Calibration Module
-----------------------------

The calibration module computes multichannel calibration curves from scanned films. 

Scanned films are automatically detected and selected, or ROIs can be drawn manually.

The lateral scanner response effect (inhomogeneous response of the scanner along the detector array) can be accounted for by creating separate calibration curves for each pixel along the array.
This requires exposing long film strips and scanning them perpendicular to the scan direction (see demonstration files). 
To account for non-flat beam profiles, a text file containing the relative beam profile shape along the film strips can be given as input to correct for non-uniform dose on the film.
Alternatively, the lateral scanner response correction can be turned off in which a single calibration curve is computed for all pixels. This simpler calibration is adequate if scanning only small films at a reproducible location on the scanner.

Features
^^^^^^^^

- Automatically loads multiple images in a folder, averages multiple copies of the same image and stacks different scans together.
- Automatically detects film position and size, and defines ROIs inside these films.
- Daily output correction
- Beam profile correction
- Lateral scanner response correction
- Save/Load LUT files
- Publish PDF report


Film-to-Dose Module
-------------------

The film-to-dose module performs optimized multichannel conversion from scanned gafchromic films to absolute dose.
It uses the optimized multichannel method from Mayer *et al* (https://doi.org/10.1118/1.3694100) and calibration curves obtained with the calibration module.

Features
^^^^^^^^

- Multiple scans of same film are loaded and averaged automatically
- Automatic film detection and crop
- Multichannel optimized conversion to absolute dose (reduced film inhomogeneities/artefacts)
- Lateral scanner response is accounted for if this feature was turned on during calibration
- Calibration curve interpolation performed by fitting either a rational function or spline curve
- Output individual channels dose (R/G/B), as well as optimized dose, mean channel dose and average dose
- Output metrics for evaluation of dose conversion quality: disturbance map, residual error, consistency map
- Publish PDF report


Dose Analysis Module
--------------------

The dose analysis module performs in-depth comparison from film dose to reference dose image from treatment planning system.

Features
^^^^^^^^

- Perform registration by identifying fiducial markers to set isocenter
- Interactive display of analysis results (gamma map, relative error, dose profiles)
- Gamma analysis: display gamma map, pass rate, histogram, pass rate vs dose bar graph, pass rate vs distance to agreement (fixed dose to agreement), pass rate vs dose to agreement (fixed distance to agreement)
- Publish PDF report
