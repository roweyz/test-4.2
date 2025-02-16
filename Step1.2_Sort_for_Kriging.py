import numpy as np
import scipy.io as sio

samplesize = 500
filename = 'sobol'
ap_step = 26 + 1
ss_step = 84 + 1
gridNumber = ap_step * ss_step

# alldata = \
#     sio.loadmat(
#         './Data_Generated/TrainingData_' + filename + '/' + str(samplesize) + '_spectral_radius_for_Kriging.mat')[
#         'data']
alldata = sio.loadmat('./Data_Presentation/500_spectral_radius_for_Kriging.mat')['data']
for i in range(0, gridNumber):
    temp = []
    for j in range(0, samplesize):
        if j == 0:
            temp = alldata[gridNumber * j + i, :]
        else:
            temp = np.vstack((temp, alldata[gridNumber * j + i, :]))

    # sio.savemat('./Data_Generated/TrainingData_' + filename + '/KrigingData' + str(i) + '.mat', {'data': temp})
    sio.savemat('./Data_Presentation/TrainingData/KrigingData' + str(i) + '.mat', {'data': temp})

print("Step1 Sample and Sort for Kriging is over")
