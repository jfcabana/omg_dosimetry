# Optimized Multichannel Gafchromic Dosimetry (OMG Dosimetry)

With OMG Dosimetry, you can easily perform film calibration, film-to-dose conversion, and dose analysis.

It is built as an expansion to Pylinac (https://pylinac.readthedocs.io/en/latest/index.html).
Demonstration files are provided for each module to get you started quickly. 
Code is heavily commented so you can follow along and andapt it for your personnal usage.


## Gafchromic calibration module

The calibration module computes multichannel calibration curves from scanned films. 

Scanned films are automatically detected and selected, or ROIs can be drawn manually.

The lateral scanner response effect (inhomogeneous response of the scanner along the detector array) is accounted for by creating separate calibration curves for each pixel along the array.
This requires exposing long film strips and scanning them perpendicular to the scan direction (see demonstration files). 
To account for non-flat beam profiles, the output from an ICProfiler acquired at the same time as film exposure can be given as input to correct for beam shape.
Alternatively, the lateral scanner response correction can be turned off, then a single calibration curve is computed for all pixels.

### Features

- Automatically loads multiple images in a folder, average multiple copies of same image and stack different scans together.
- Automatically detect film strips position and size, and define ROIs inside these film strips.
- Daily output correction
- Beam profile correction
- Lateral scanner response correction
- Save/Load LUt files
- Publish PDF report


## Film-to-dose module

The film-to-dose module performs optimized multichannel conversion from scanned gafchromic films to absolute dose.
It uses the optimized multichannel method from Mayer *et al* (https://doi.org/10.1118/1.3694100) and calibration curves obtained with the calibration module.

### Features

- Multiple scans of same film are loaded and averaged automatically
- Automatic film detection and crop
- Multichannel optimized conversion to absolute dose (reduced film inhomogeneities/artefacts)
- Lateral scanner response is accounted for if this feature was turned on during calibration
- Calibration curves interpolation performed by fitting either a rational function or spline curve
- Output individual channels dose (R/G/B), as well as optimized dose, mean channel dose and average dose
- Output metrics for evaluation of dose conversion quality: disturbance map, residual error, consistency map
- Publish PDF report


## Dose analysis module

The dose analysis module performs in-depth comparison from film dose to reference dose image from treatment planning system.

### Features

- Perform registration by identifying fiducial markers to set isocenter
- Interactive display of analysis results (gamma map, relative error, dose profiles)
- Gammap analysis: display gamma map, pass rate, histogram, pass rate vs dose bar graph, pass rate vs distance to agreement (fixed dose to agreement), pass rate vs dose to agreement (fixed distance to agreement)
- Publish PDF report
