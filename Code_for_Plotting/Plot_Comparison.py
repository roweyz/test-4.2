# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.interpolate import make_interp_spline

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Calibri']

plt.rcParams['font.weight'] = 'normal'  # light normal heavy bold

'''------------------------------data loading-----------------------------'''
'load david SLD'
lobe_x = sio.loadmat('../Data_Presentation/David_lobe.mat')['x'][0]
lobe_y = sio.loadmat('../Data_Presentation/David_lobe.mat')['y'][0]
lobe_david = np.vstack((lobe_x, lobe_y)).T
lobe_david = lobe_david[np.argsort(lobe_david[:, 0])]

'load BSLD'
SS = sio.loadmat('../Data_Presentation/SSGrid.mat')['data']
AP = sio.loadmat('../Data_Presentation/APGrid.mat')['data']
PRO = sio.loadmat('../Data_Presentation/ProbabilityGrid.mat')['data']

'load Kriging SLD'

SS_kriging = sio.loadmat('../Data_Presentation/SS_Kriging.mat')['SS']
AP_kriging = sio.loadmat('../Data_Presentation/AP_Kriging.mat')['AP']
EI_kriging = sio.loadmat('../Data_Presentation/EI_Kriging.mat')['lambda']

'''------------------------------data loading-----------------------------'''
'wide range'
fig1, ax1 = plt.subplots(figsize=(10, 7))
linewidth = 1.5
'SLD'
'Kriging SLD'
cs1 = ax1.contour(SS_kriging, AP_kriging * 1000, EI_kriging, [1.0], colors='r', linewidths=linewidth)
'david'
plt.plot(lobe_david[:, 0], lobe_david[:, 1] * 1000, 'k-', linewidth=linewidth)
'BSLD'
cs2 = ax1.contour(SS, AP, PRO, [0.5], colors='b', linewidths=linewidth)

'EXP data'
kriging_exp = np.array(pd.read_csv('../Data_Essential/MTM_Exp.csv', sep=','))
Stable = []
Unstable = []
Margin = []
for i in range(kriging_exp.shape[0]):
    if int(kriging_exp[i, 2]) == 1:
        Unstable.append(np.array(kriging_exp[i]))
    elif int(kriging_exp[i, 2]) == 0:
        Stable.append(np.array(kriging_exp[i]))
    else:
        Margin.append(np.array(kriging_exp[i]))

Stable = np.array(Stable)
Unstable = np.array(Unstable)
Margin = np.array(Margin)

'''------------------------------legend plotting-----------------------------'''
'SLD labels'
plt.plot([5000, 5000], [0.1, 0.1], 'r', label='Our method')
plt.plot([5000, 5000], [0.1, 0.1], 'b', label='Chen et al.')
plt.plot(lobe_david[:, 0], lobe_david[:, 1] * 1000, 'k-', label='Hajdu et al.')

'Exp labels'
plt.scatter(Stable[:, 0] * 10000, Stable[:, 1], c='k', marker='o', s=50, label='Stable')
plt.scatter(Unstable[:, 0] * 10000, Unstable[:, 1], c='k', marker='x', s=50, label='Unstable')
plt.scatter(Margin[:, 0] * 10000, Margin[:, 1], c='w', marker='^', edgecolors='k', s=50, label='Margin')


'''------------------------------Figures setting-----------------------------'''
fontsize_axis = 20
fontsize_legend = 15
fontsize_tick = 15
fontsize_title = 20

ax1.set_xlabel('Spindle speed (rev/min)', fontsize=fontsize_axis)
ax1.set_ylabel('Axial depth (mm)', fontsize=fontsize_axis)

ax1.tick_params(axis='x', labelsize=fontsize_tick)
ax1.tick_params(axis='y', labelsize=fontsize_tick)

plt.tight_layout()
ax = plt.gca()
plt.legend(loc='upper left', fontsize=fontsize_legend, ncol=2)

plt.savefig('../Figures/Fig9_Comparison.png', dpi=300, bbox_inches='tight')

'''------------------------------'high speed representation-----------------------------'''
'''------------------------------boundary points of stable interpolation----------------------------'''

sorted_data = kriging_exp[np.lexsort([kriging_exp[:, 2], kriging_exp[:, 1], kriging_exp[:, 0]]), :]
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

plotx_stable = []
ploty_stable = []

for i in range(len(boundary_stable)):
    plotx_stable.append(boundary_stable[i][0] * 10000)
    ploty_stable.append(boundary_stable[i][1])

plotx_stable_smooth = np.linspace(plotx_stable[0], plotx_stable[len(plotx_stable) - 1], 300)
ploty_stable_smooth = make_interp_spline(plotx_stable, ploty_stable)(plotx_stable_smooth)

'''------------------------------plotting-----------------------------'''
fig2, ax2 = plt.subplots(figsize=(7, 7))
linewidth = 1.5
plt.scatter(Stable[:, 0] * 10000, Stable[:, 1], c='k', marker='o', s=50)
plt.scatter(Unstable[:, 0] * 10000, Unstable[:, 1], c='k', marker='x', s=50)
plt.scatter(Margin[:, 0] * 10000, Margin[:, 1], c='w', marker='^', edgecolors='k', s=50)

'Kriging SLD'
ax2.contour(SS_kriging, AP_kriging * 1000, EI_kriging, [1.0], colors='r', linewidths=linewidth)
'david'
plt.plot(lobe_david[:, 0], lobe_david[:, 1] * 1000, 'k-', linewidth=linewidth)
'BSLD'
ax2.contour(SS, AP, PRO, [0.5], colors='b', linewidths=linewidth)

plt.plot(plotx_stable_smooth, ploty_stable_smooth, color='grey', )
plt.fill_between(plotx_stable_smooth, 0, ploty_stable_smooth, facecolor='grey', alpha=0.3,
                 label='experimentally suggested stable area')

fontsize_axis = 20
fontsize_legend = 15
fontsize_tick = 15
fontsize_title = 20

plt.xlim((8800, 13000))
plt.ylim((0.25, 3.5))

ax2.set_xticks(np.arange(9000, 14000, 1000))
ax2.tick_params(axis='x', labelsize=fontsize_tick)
ax2.tick_params(axis='y', labelsize=fontsize_tick)

plt.tight_layout()
ax = plt.gca()
plt.legend(loc='upper left', fontsize=fontsize_legend, ncol=1)

plt.savefig('../Figures/Fig9_Comparison_High_Speed.png', dpi=300, bbox_inches='tight')
plt.show()
