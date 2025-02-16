# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Calibri']

plt.rcParams['font.weight'] = 'normal'  # light normal heavy bold

BSLDei = sio.loadmat('./Data_Presentation/ProbabilityGrid.mat')['data']
KSLDei = sio.loadmat('./Data_Presentation/EI_Kriging.mat')['lambda']
exp_data = kriging_exp = np.array(pd.read_csv('./Data_Essential/MTM_Exp.csv', sep=','))

sorted_data = exp_data[np.lexsort([exp_data[:, 2], exp_data[:, 1], exp_data[:, 0]]), :]

'''------------------------------Boundary of EXP----------------------------'''

boundary_stable = []
boundary_unstable = []
boundary_margin = []

i = 0
while i < sorted_data.shape[0]:
    ss1 = sorted_data[i][0]
    index = 0
    l = []
    l.append(sorted_data[i])
    while True:
        index = index + 1
        if i + index >= sorted_data.shape[0]:
            break
        if (sorted_data[i + index][0] != ss1):
            break
        else:
            l.append(sorted_data[i + index])
            continue
    i = i + index
    l0 = []
    l1 = []
    l2 = []
    for j in range(len(l)):
        if l[j][2] == 0:
            l0.append(l[j])
        elif l[j][2] == 1:
            l1.append(l[j])
        else:
            l2.append(l[j])
    if len(l0) != 0:
        boundary_stable.append(l0.pop())
    if len(l1) != 0:
        boundary_unstable.append(l1[0])
    if len(l2) != 0:
        boundary_margin.append(l2[0])

boundary_stable = np.array(boundary_stable)
boundary_stable = boundary_stable[:, 0:2]
boundary_unstable = np.array(boundary_unstable)
boundary_unstable = boundary_unstable[:, 0:2]
boundary_margin = np.array(boundary_margin)
boundary_margin = boundary_margin[:, 0:2]

'------------------------------Boundary of SLD----------------------------'

boundary_KSLD = sio.loadmat('./Data_Presentation/boundary_KSLD.mat')['BoundaryKSLD']
boundary_BSLD = sio.loadmat('./Data_Presentation/boundary_BSLD.mat')['BoundaryBSLD']
lobe_x = sio.loadmat('./Data_Presentation/David_lobe.mat')['x'][0]
lobe_y = sio.loadmat('./Data_Presentation/David_lobe.mat')['y'][0]
lobe_david = np.vstack((lobe_x, lobe_y)).T
lobe_david = lobe_david[np.argsort(lobe_david[:, 0])]

boundary_KSLD = np.array(sorted(boundary_KSLD, key=lambda x: x[0]))
boundary_david = np.zeros((lobe_david.shape[0], 2))
# print(len(boundary_KSLD), len(boundary_BSLD), len(boundary_david))

for i in range(lobe_david.shape[0]):
    boundary_david[i][0] = lobe_david[i][0] / 10000
    boundary_david[i][1] = lobe_david[i][1] * 1000

'---------------------------------------------------------------------'
# x axis (1e3 rpm)/y axis (mm)
for i in range(boundary_BSLD.shape[0]):
    boundary_BSLD[i][0] = boundary_BSLD[i][0] * 10

for i in range(boundary_KSLD.shape[0]):
    boundary_KSLD[i][0] = boundary_KSLD[i][0] * 10

for i in range(boundary_david.shape[0]):
    boundary_david[i][0] = boundary_david[i][0] * 10

for i in range(boundary_stable.shape[0]):
    boundary_stable[i][0] = boundary_stable[i][0] * 10

for i in range(boundary_unstable.shape[0]):
    boundary_unstable[i][0] = boundary_unstable[i][0] * 10
'------------------------------SLD resample---------------------------'

resample_size = 26
# resample_size = 40
boundary_KSLD = boundary_KSLD[::round(len(boundary_KSLD) / resample_size)]
boundary_BSLD = boundary_BSLD[::round(len(boundary_BSLD) / resample_size)]
boundary_david = boundary_david[::round(len(boundary_david) / resample_size)]

fig, ax = plt.subplots(figsize=(10, 5))
plt.scatter(boundary_stable[:, 0], boundary_stable[:, 1], color='k', marker='o', linewidths=1.5, s=25, label='Stable')
plt.scatter(boundary_unstable[:, 0], boundary_unstable[:, 1], color='k', marker='x', linewidths=1.5, s=25,
            label='Unstable')
plt.scatter(boundary_KSLD[:, 0], boundary_KSLD[:, 1], color='r', marker='o', linewidths=1.5, s=1, label='KSLD')
plt.scatter(boundary_BSLD[:, 0], boundary_BSLD[:, 1], color='b', marker='o', linewidths=1.5, s=1, label='BSLD')
plt.scatter(boundary_david[:, 0], boundary_david[:, 1], color='k', marker='o', linewidths=1.5, s=0.1, label='david')
plt.title('Resampled boundary points', fontsize=15, color='k', fontweight='normal', loc='center')
plt.legend(fontsize=10, loc='upper left', ncol=1)
plt.show()
'------------------------------Demonstration plotting---------------------------'
boundary_KSLD = boundary_KSLD[1:, :]

# plt.scatter(boundary_stable[:, 0], boundary_stable[:, 1], color='b', marker='o', linewidths=1.5, s = 25, label='Stable')
# plt.scatter(boundary_unstable[:, 0], boundary_unstable[:, 1], color='k', marker='x', linewidths=1.5, s = 25, label='Unstable')
# plt.scatter(boundary_KSLD[:, 0], boundary_KSLD[:, 1], color='r', marker='o', linewidths=1.5, s = 10, label = 'KSLD')
# plt.scatter(boundary_BSLD[:, 0], boundary_BSLD[:, 1], color='b', marker='o', linewidths=1.5, s = 10, label = 'BSLD')

'import fastDTW'

distance_stable_BSLD, path_stable_BSLD = fastdtw(boundary_stable, boundary_BSLD, dist=euclidean)
distance_stable_KSLD, path_stable_KSLD = fastdtw(boundary_stable, boundary_KSLD, dist=euclidean)
distance_stable_david, path_stable_david = fastdtw(boundary_stable, boundary_david, dist=euclidean)

'Calcu verification'
distance_unstable_BSLD, path_unstable_BSLD = fastdtw(boundary_unstable, boundary_BSLD, dist=euclidean)
distance_unstable_KSLD, path_unstable_KSLD = fastdtw(boundary_unstable, boundary_KSLD, dist=euclidean)
distance_unstable_david, path_unstable_david = fastdtw(boundary_unstable, boundary_david, dist=euclidean)

print('\nfinal dtw\n')
print('KSLD:', (distance_stable_KSLD + distance_unstable_KSLD) / 2)
print('BSLD:', (distance_stable_BSLD + distance_unstable_BSLD) / 2)
print('david:', (distance_stable_david + distance_unstable_david) / 2)
