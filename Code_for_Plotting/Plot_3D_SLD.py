import matplotlib.pyplot as plt
import numpy as np
import scipy.io

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Calibri']

plt.rcParams['font.weight'] = 'normal'  # light normal heavy bold

'''------------------------------SLD data loading----------------------------'''
data = scipy.io.loadmat('../Data_Presentation/spectral_radius_for_3D_SLD.mat')['data']

ss = np.linspace(4800, 13200, 101)
dc = np.linspace(0, 0.004, 51)
dc = dc * 1000
Nx = ss.size
Ny = dc.size

ei = np.zeros((Nx, Ny))

for i in range(Nx):
    for j in range(Ny):
        ei[i, j] = data[(i) * 51 + j, 10]

'''------------------------------SLD plotting---------------------------'''
SS, DC = np.meshgrid(ss, dc)
SS = SS.T
DC = DC.T

'3D SLD surface'
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(SS, DC, ei, cmap='viridis')

'eigen plane'
Z = np.ones_like(SS) * 1
plane = ax.plot_surface(SS, DC, Z, color='grey', alpha=0.4)

'2D SLD contour'
Z_intersection = np.interp(1, DC[:, 0], ei[:, 0])
contour = ax.contour(SS, DC, ei, levels=[Z_intersection], offset=1, colors='white', linewidths=2, linestyles='solid',
                     zdir='z')

'''------------------------------Figures setting-----------------------------'''
fontsize_axis = 15
fontsize_tick = 10
fontsize_title = 20

ax.set_box_aspect([10, 6, 3])  # length width height

ax.set_xlabel('spindle speed (rpm)', labelpad=15, fontsize=fontsize_axis)
ax.set_ylabel('depth of cut (mm)', labelpad=0, fontsize=fontsize_axis)

ax.zaxis.set_rotate_label(False)
ax.set_zlabel('$\lambda$', labelpad=0, fontsize=fontsize_axis)

ax.set_xlim(5000, 13000)
ax.set_xticks([4000, 6000, 8000, 10000, 12000])
ax.set_ylim(0, 4)
ax.set_yticks([1, 3])
ax.set_zticks([1, 3])

ax.tick_params(axis='both', which='major', labelsize=fontsize_tick)
fig.set_size_inches(10, 6)
plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)
plt.savefig('../Figures/Fig.1_3D_SLD.png', dpi=300, bbox_inches='tight')
plt.show()
