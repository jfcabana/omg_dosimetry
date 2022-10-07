# -*- coding: utf-8 -*-
#%% Test npgamma

import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#matplotlib inline

from npgamma import calc_gamma

with open('S:/Python/EBT3/Tools/npgamma/data_unequal_grid.yml', 'r') as file:
    data = yaml.load(file)
    
x_reference = data['x_mephisto']
y_reference = data['d_mephisto']
dose_reference = data['mephisto_dose_grid']

x_evaluation = data['x_monaco']
y_evaluation = data['d_monaco']
dose_evaluation = data['monaco_dose_grid']

coords_reference = (y_reference, x_reference)
coords_evaluation = (y_evaluation, x_evaluation)

distance_threshold = 2
distance_step_size = distance_threshold / 10

dose_threshold = 0.02 * np.max(dose_evaluation)

lower_dose_cutoff = np.max(dose_evaluation) * 0.2

gamma = calc_gamma(
    coords_reference, dose_reference,
    coords_evaluation, dose_evaluation,
    distance_threshold, dose_threshold,
    lower_dose_cutoff=lower_dose_cutoff, 
    distance_step_size=distance_step_size,
    maximum_test_distance=np.inf)

valid_gamma = gamma[~np.isnan(gamma)]
valid_gamma[valid_gamma > 2] = 2

plt.hist(valid_gamma, 30);
plt.xlim([0,2])


#%% test pygamma
import numpy
import pylab

from algorithms import gamma_evaluation 

# Reference data with (2, 1) mm resolution
reference = numpy.random.random((128, 256))
#reference = numpy.abs(reference)
reference /= reference.max()
reference *= 100
reference -= 50

# Sample data with a %3 shift on the reference
sample = reference * 1.03

# Perform gamma evaluation at 4mm, 2%, resoution x=2, y=1
gamma_map = gamma_evaluation(sample, reference, 4., 2., (1, 1), signed=True)

pylab.imshow(gamma_map, cmap='RdBu_r', aspect=2, vmin=-2, vmax=2)
pylab.colorbar()
pylab.show()

