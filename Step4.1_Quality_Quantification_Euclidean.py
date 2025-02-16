# -*- coding: utf-8 -*-

import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Calibri']

plt.rcParams['font.weight'] = 'normal'  # light normal heavy bold

'''--------------------------------------------------Part I---------------------------------------------'''
'''------------------------------Boundary of EXP----------------------------'''

boundary_stable = []
boundary_unstable = []
boundary_margin = []

exp_data = kriging_exp = np.array(pd.read_csv('./Data_Essential/MTM_Exp.csv', sep=','))

sorted_data = exp_data[np.lexsort([exp_data[:, 2], exp_data[:, 1], exp_data[:, 0]]), :]
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
boundary_unstable = np.array(boundary_unstable)
boundary_margin = np.array(boundary_margin)

'''------------------------------Boundary of SLD----------------------------'''

boundary_KSLD = sio.loadmat('./Data_Presentation/boundary_KSLD.mat')['BoundaryKSLD']
boundary_BSLD = sio.loadmat('./Data_Presentation/boundary_BSLD.mat')['BoundaryBSLD']
lobe_x = sio.loadmat('./Data_Presentation/David_lobe.mat')['x'][0]
lobe_y = sio.loadmat('./Data_Presentation/David_lobe.mat')['y'][0]
lobe_david = np.vstack((lobe_x, lobe_y)).T
lobe_david = lobe_david[np.argsort(lobe_david[:, 0])]

boundary_KSLD = np.array(sorted(boundary_KSLD, key=lambda x: x[0]))

boundary_david = np.zeros((lobe_david.shape[0], 2))
for i in range(lobe_david.shape[0]):
    boundary_david[i][0] = lobe_david[i][0] / 10000
    boundary_david[i][1] = lobe_david[i][1] * 1000

'figure 1'
'''------------------------------Discretized boundary plotting----------------------------'''

fig, ax = plt.subplots(figsize=(10, 5))
plt.scatter(boundary_stable[:, 0], boundary_stable[:, 1], color='k', marker='o', linewidths=1.5, s=25, label='Stable')
plt.scatter(boundary_unstable[:, 0], boundary_unstable[:, 1], color='k', marker='x', linewidths=1.5, s=25,
            label='Unstable')
plt.scatter(boundary_margin[:, 0], boundary_margin[:, 1], color='k', marker='^', linewidths=1.5, s=25, label='Margin')
plt.scatter(boundary_KSLD[:, 0], boundary_KSLD[:, 1], color='r', marker='o', linewidths=1.5, s=1, label='KSLD')
plt.scatter(boundary_BSLD[:, 0], boundary_BSLD[:, 1], color='b', marker='o', linewidths=1.5, s=1, label='BSLD')
plt.scatter(boundary_david[:, 0], boundary_david[:, 1], color='k', marker='o', linewidths=1.5, s=0.1, label='david')

plt.xticks(np.arange(0.45, 1.45, 0.15))
plt.yticks(np.arange(0, 4, 0.5))
plt.title('Discretized boundary', fontsize=15, color='k', fontweight='normal', loc='center')
plt.legend(fontsize=10, loc='upper left', ncol=2)

'''--------------------------------------------------Part II---------------------------------------------'''
'''------------------------------Euclidean calculation---------------------------'''

ss_stable = boundary_stable[:, 0]
ss_unstable = boundary_unstable[:, 0]
ss_KSLD = boundary_KSLD[:, 0]
ss_BSLD = boundary_BSLD[:, 0]
ss_david = boundary_david[:, 0]

nearest_indices_KSLD = []
nearest_indices_BSLD = []
nearest_indices_david = []

# nearest index of SLD can be calculated based on either ss_stable or ss_ubstable
for a in ss_stable:
    nearest_index_KSLD = np.abs(ss_KSLD - a).argmin()
    nearest_index_BSLD = np.abs(ss_BSLD - a).argmin()
    nearest_index_david = np.abs(ss_david - a).argmin()
    nearest_indices_KSLD.append(nearest_index_KSLD)
    nearest_indices_BSLD.append(nearest_index_BSLD)
    nearest_indices_david.append(nearest_index_david)

# for a in ss_unstable:
#     nearest_index_KSLD = np.abs(ss_KSLD - a).argmin()
#     nearest_index_BSLD = np.abs(ss_BSLD - a).argmin()
#     nearest_index_david = np.abs(ss_david - a).argmin()
#     nearest_indices_KSLD.append(nearest_index_KSLD)
#     nearest_indices_BSLD.append(nearest_index_BSLD)
#     nearest_indices_david.append(nearest_index_david)

nearest_KSLD = boundary_KSLD[nearest_indices_KSLD, :]
nearest_BSLD = boundary_BSLD[nearest_indices_BSLD, :]
nearest_david = boundary_david[nearest_indices_david, :]

'figure 2'
fig2, ax2 = plt.subplots(figsize=(10, 5))
plt.scatter(nearest_KSLD[:, 0], nearest_KSLD[:, 1], color='r', marker='+', s=10, label='KSLD')
plt.scatter(nearest_BSLD[:, 0], nearest_BSLD[:, 1], color='b', marker='+', s=10, label='BSLD')
plt.scatter(nearest_david[:, 0], nearest_david[:, 1], color='k', marker='+', s=10, label='david')


plt.scatter(boundary_unstable[:, 0], boundary_unstable[:, 1], color='k', marker='x', s=20, label='unstable')

for value in ss_unstable:
    plt.axvline(x=value, linestyle='--', color='grey', alpha=0.4)

plt.xticks(np.arange(0.45, 1.45, 0.15))
plt.yticks(np.arange(0, 4, 0.5))
plt.title('Nearest boundary point', fontsize=15, color='k', fontweight='normal', loc='center')
plt.legend(fontsize=10, loc='upper left', ncol=1)
plt.show()

'''--------------------------------------------------Part III---------------------------------------------'''
'''------------------------------Euclidean calculation---------------------------'''
dc_nearest_KSLD = nearest_KSLD[:, 1]
dc_nearest_BSLD = nearest_BSLD[:, 1]
dc_nearest_david = nearest_david[:, 1]
dc_stable = boundary_stable[:, 1]
dc_unstable = boundary_unstable[:, 1]

error_stable_KSLD = dc_nearest_KSLD - dc_stable
error_stable_BSLD = dc_nearest_BSLD - dc_stable
error_stable_david = dc_nearest_david - dc_stable

error_unstable_KSLD = dc_nearest_KSLD - dc_unstable
error_unstable_BSLD = dc_nearest_BSLD - dc_unstable
error_unstable_david = dc_nearest_david - dc_unstable

absolute_error_KSLD = (np.sum(np.abs(error_stable_KSLD)) + np.sum(np.abs(error_unstable_KSLD))) / 2
absolute_error_BSLD = (np.sum(np.abs(error_stable_BSLD)) + np.sum(np.abs(error_unstable_BSLD))) / 2
absolute_error_david = (np.sum(np.abs(error_stable_david)) + np.sum(np.abs(error_unstable_david))) / 2

print('final Euclidean')
print('absolute error KSLD:', absolute_error_KSLD)
print('absolute error BSLD:', absolute_error_BSLD)
print('absolute error david:', absolute_error_david)

