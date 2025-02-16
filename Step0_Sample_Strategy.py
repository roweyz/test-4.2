# -*- coding: utf-8 -*-

import os

import numpy as np
from scipy.stats import qmc, norm

'''------------------------------sample for kriging----------------------------'''
n_dim = 8
n_samples = 256

mu = [0.7827, 0.7528, 0.0184, 0.0186, 6.56, 4.88, 10.95, 1.76]
sigma = [0.025, 0.025, 0.001, 0.001, 0.15, 0.1, 0.2, 0.05]

'Sobol'
sampler = qmc.Sobol(d=n_dim)
sobol_samples = sampler.random(n=n_samples)

normal_samples = norm.ppf(sobol_samples)
data_sobol = np.multiply(normal_samples, sigma) + mu
data_sobol[:, 0:2] = data_sobol[:, 0:2] * 1000

data_sobol_max = np.zeros(shape=n_dim)
data_sobol_min = np.zeros(shape=n_dim)
for i in range(data_sobol.shape[1]):
    data_sobol_max[i] = np.double(max(data_sobol[:, i]))
    data_sobol_min[i] = np.double(min(data_sobol[:, i]))

# print("sample quality check:")
# print("plz compare to OMA function range")

print("sobol max:\n", data_sobol_max)
print("sobol min:\n", data_sobol_min)

'data save'
directory = './Data_Essential'
file_name1 = 'sobol.npy'

file_path1 = os.path.join(directory, file_name1)
np.save(file_path1, data_sobol)
