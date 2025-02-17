import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import scipy.stats as stats
from scipy.interpolate import interp1d

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Calibri']

plt.rcParams['font.weight'] = 'normal'  # light normal heavy bold

# SS_kriging = scipy.io.loadmat('../Data_Essential/SS_Kriging.mat')['SS']
# AP_kriging = scipy.io.loadmat('../Data_Essential/AP_Kriging.mat')['AP']
# EI_kriging = scipy.io.loadmat('../Data_Essential//EI_Kriging.mat')['lambda']
# STD_kriging = scipy.io.loadmat('../Data_Essential//STD_Kriging.mat')['standard']
SS_kriging = scipy.io.loadmat('../Data_Presentation/SS_Kriging.mat')['SS']
AP_kriging = scipy.io.loadmat('../Data_Presentation/AP_Kriging.mat')['AP']
# EI_kriging = scipy.io.loadmat('../Data_Presentation//EI_Kriging.mat')['lambda']
# STD_kriging = scipy.io.loadmat('../Data_Presentation//STD_Kriging.mat')['standard']
EI_kriging = scipy.io.loadmat('../Data_Presentation//EI.mat')['lambda']
STD_kriging = scipy.io.loadmat('../Data_Presentation//STD.mat')['standard']

'''------------------------------confidence interval selection------------------------------'''

df = 500  # degree of freedom for t distribution
alpha = 0.15  # confidence level
t_upper_quantile = stats.t.ppf(1 - alpha, df)  # upper quantile

EI_kriging_lower = EI_kriging + t_upper_quantile * STD_kriging
EI_kriging_upper = EI_kriging - t_upper_quantile * STD_kriging

'''------------------------------confidence interval plot------------------------------'''
fig, ax = plt.subplots(figsize=(10.8, 6))

cs1 = ax.contour(SS_kriging, AP_kriging * 1000, EI_kriging, [1, 10], colors='r')
cs2 = ax.contour(SS_kriging, AP_kriging * 1000, EI_kriging_lower, [1, 10], colors='darkorange', alpha=0.5)
cs3 = ax.contour(SS_kriging, AP_kriging * 1000, EI_kriging_upper, [1, 10], colors='red', alpha=0.5)

'''------------------------------Cutting Test Result------------------------------'''
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

'''------------------------------Fill between------------------------------'''
'boundary Kstable'
for i, collection in enumerate(cs1.collections):
    paths = collection.get_paths()
    for path in paths:
        boundary_KSLD = path.vertices
        x_points, y_points = boundary_KSLD[:, 0], boundary_KSLD[:, 1]

ss_boundary_KSLD = boundary_KSLD[:, 0]
dc_boundary_KSLD = boundary_KSLD[:, 1]

'boundary Kstable'
for i, collection in enumerate(cs2.collections):
    paths = collection.get_paths()
    for path in paths:
        boundary_Kstable = path.vertices
        x_points, y_points = boundary_Kstable[:, 0], boundary_Kstable[:, 1]

ss_boundary_Kstable = boundary_Kstable[:, 0]
dc_boundary_Kstable = boundary_Kstable[:, 1]

'boundary Kupper'
for i, collection in enumerate(cs3.collections):
    paths = collection.get_paths()
    for path in paths:
        boundary_Kupper = path.vertices
        x_points, y_points = boundary_Kupper[:, 0], boundary_Kupper[:, 1]

ss_boundary_Kupper = boundary_Kupper[:, 0]
dc_boundary_Kupper = boundary_Kupper[:, 1]

ssinterval = np.arange(1.32, 0.48, -0.01)
ssinterval = np.append(ssinterval, 0.48)
ssinterval_fillbetween = ssinterval * 10000
boundary_Kstable_fillbetween = np.zeros([len(ssinterval), 2])
boundary_KSLD_fillbetween = np.zeros([len(ssinterval), 2])
boundary_Kupper_fillbetween = np.zeros([len(ssinterval), 2])

for i in range(len(ssinterval)):
    temp = ssinterval[i]
    j = 0
    while j < len(boundary_Kstable):
        if abs(boundary_Kstable[j, 0] - temp) < 1e-5:
            boundary_Kstable_fillbetween[i, 0] = boundary_Kstable[j, 0]
            boundary_Kstable_fillbetween[i, 1] = boundary_Kstable[j, 1]
            break
        else:
            j = j + 1
    k = 0
    while k < len(boundary_KSLD):
        if abs(boundary_KSLD[k, 0] - temp) < 1e-5:
            boundary_KSLD_fillbetween[i, 0] = boundary_KSLD[k, 0]
            boundary_KSLD_fillbetween[i, 1] = boundary_KSLD[k, 1]
            break
        else:
            k = k + 1
    l = 0

    while l < len(boundary_Kupper):
        if abs(boundary_Kupper[l, 0] - temp) < 1e-5:
            boundary_Kupper_fillbetween[i, 0] = boundary_Kupper[l, 0]
            boundary_Kupper_fillbetween[i, 1] = boundary_Kupper[l, 1]
            break
        else:
            l = l + 1

dc_boundary_Kstable_fillbetween = boundary_Kstable_fillbetween[:, 1]
dc_boundary_KSLD_fillbetween = boundary_KSLD_fillbetween[:, 1]
dc_boundary_Kupper_fillbetween = boundary_Kupper_fillbetween[:, 1]

'''------------------------------legend plotting-----------------------------'''
'SLD labels'
plt.plot([5000, 5000], [0.1, 0.1], 'r', label='Our method')
plt.plot([5000, 5000], [0.1, 0.1], 'darkorange', alpha=0.5, label='lower boundary')
plt.plot([5000, 5000], [0.1, 0.1], 'red', alpha=0.5, label='upper boundary')

'Exp labels'
plt.scatter(Stable[:, 0] * 10000, Stable[:, 1], c='k', marker='o', s=50, label='Stable')
plt.scatter(Unstable[:, 0] * 10000, Unstable[:, 1], c='k', marker='x', s=50, label='Unstable')
plt.scatter(Margin[:, 0] * 10000, Margin[:, 1], c='w', marker='^', edgecolors='k', s=50, label='Margin')

'Fillbetween labels'
plt.fill_between(ss_boundary_Kstable, 0, dc_boundary_Kstable, facecolor='darkorange', alpha=0.15,
                 label='robust stable area')
interp_func = interp1d(ss_boundary_Kstable, dc_boundary_Kstable, bounds_error=False, fill_value="extrapolate")
dc_boundary_Kstable_interpolated = interp_func(ss_boundary_Kupper)
plt.fill_between(ss_boundary_Kupper, dc_boundary_Kstable_interpolated, dc_boundary_Kupper, facecolor='red', alpha=0.15,
                 label='confidence interval({:.0f}%)'.format((1 - alpha) * 100))

'Improvements labels'
'85%'
# Improvements_85 = np.array(
#     [(6200, 0.5), (6400, 0.5), (7600, 1), (8000, 0.75), (7800, 1), (8400, 0.5), (8800, 0.5),
#      (11500, 1.5), (11500, 1.75)])
Improvements_85 = np.array(
    [(5000, 0.5), (5200, 0.5), (6200, 0.5), (6400, 0.5), (8000, 0.75), (7800, 1), (8400, 0.5), (8800, 0.5),
     (11500, 1.5), (11500, 1.75)])

plt.scatter(Improvements_85[:, 0], Improvements_85[:, 1], linewidths=1.5, edgecolors='b', c='None', marker='s',
            s=140, label='Improvements')
'''------------------------------Figures setting-----------------------------'''

fontsize_axis = 20
fontsize_legend = 13
fontsize_tick = 15
fontsize_title = 20

ax.set_xlabel('Spindle speed (rev/min)', fontsize=fontsize_axis)
ax.set_ylabel('Axial depth (mm)', fontsize=fontsize_axis)

ax.tick_params(axis='x', labelsize=fontsize_tick)
ax.tick_params(axis='y', labelsize=fontsize_tick)

plt.tight_layout()
ax = plt.gca()
plt.legend(loc='upper right', fontsize=fontsize_legend, ncol=3)

# ax.set_title(r'$\gamma$ = {:.2f}, confidence interval = {:.0f}%'.format(alpha, (1 - alpha) * 100),
#              fontsize=fontsize_title)

# filename = '../Figures/Fig8_{:.0f}%CI.png'.format((1 - alpha) * 100)
filename = '../Figures/Fig8_Confidence_Interval.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')

plt.show()
