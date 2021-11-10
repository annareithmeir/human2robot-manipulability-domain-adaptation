import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
from numpy import genfromtxt
import numpy as np
from scipy import linalg
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import get_cov_ellipsoid


colors=['green', 'blue', 'orange', 'red', 'purple']



###########################
### plot demonstrations ###

n_demos=4
n_points=20
#n_points=400
scaling_factor=1e-1
plot_every_nth = 5

colors=['green', 'blue', 'orange', 'red', 'purple']

fig = plt.figure()
plt.subplot(1, 2, 1)
ax = plt.axes(projection='3d')
plt.title('Demonstrations and GMR results')

#data_path = "/home/nnrthmr/Desktop/master-thesis/promps-code/tum_tuda_project/recorded_data/Session_25_06_2021/run1.csv"
#data_path = "../data/demos/dummyTrajectories.csv"

data_path = "../data/demos/trajectories.csv"


data = pd.read_csv(data_path, sep=",")
xdata= np.array(data['EE_x'])[:n_demos*n_points]
ydata= np.array(data['EE_y'])[:n_demos*n_points]
zdata= np.array(data['EE_z'])[:n_demos*n_points]
#xdata= np.array(data['panda_left_EE_x'])
#ydata= np.array(data['panda_left_EE_y'])
#zdata= np.array(data['panda_left_EE_z'])
ax.scatter3D(xdata, ydata, zdata, c='grey', alpha=0.4)
ax.scatter3D(0,0,0, c='red', marker='x', s=100)


'''
### plot demonstration manipulabilities ###

#filename_manip = "../data/demos/tum_tuda/translationManip3d.csv"
filename_manip = "../data/demos/translationManip3d.csv"
manip_tmp = genfromtxt(filename_manip, delimiter=',')
manip_tmp=manip_tmp[1:,:]
manip=list()

for i in np.arange(0, manip_tmp.shape[0]):
    manip.append(scaling_factor*manip_tmp[i,1:].reshape(3,3))

manip=manip[:n_demos*n_points]

for i in np.arange(0,len(manip),plot_every_nth):
    m_i = manip[i]
    X2,Y2,Z2 = get_cov_ellipsoid(m_i, [xdata[i],ydata[i],zdata[i]], 1)
    ax.plot_wireframe(X2,Y2,Z2, color='grey', alpha=0.05)


### plot GMR results ###

#filename_mu = "../data/expData3d.csv"
filename_sigma= "../data/results/xhat.csv"
filename_mu = "../data/results/xd.csv"

mu_tmp = genfromtxt(filename_mu, delimiter=',')
sigma_tmp = genfromtxt(filename_sigma, delimiter=',')
sigma_tmp = scaling_factor*sigma_tmp

mu=list()
sigma=list()

for i in np.arange(n_points):
    mu.append(mu_tmp[:,i])
    sigma.append(sigma_tmp[i,:].reshape(3,3))

for i in np.arange(n_points):
    mu_i = mu[i]
    ax.scatter(mu_i[0],mu_i[1],mu_i[2], color='blue')


for i in np.arange(0,n_points,plot_every_nth):
    mu_i = mu[i]
    sigma_i=sigma[i]
    X2,Y2,Z2 = get_cov_ellipsoid(sigma_i, mu_i, 1)
    ax.plot_wireframe(X2,Y2,Z2, color='blue', alpha=0.1)

'''

import matplotlib.patches as mpatches

red_patch = mpatches.Patch(color='red', label='Robot base location')
blue_patch = mpatches.Patch(color='blue', label='Learned')
grey_patch = mpatches.Patch(color='grey', label='Demonstrated')

plt.legend(handles=[red_patch, blue_patch, grey_patch])
plt.xlim(-0.5, 0.5)
plt.ylim(-1, 1)
ax.set_zlim(0., 1)
plt.show()






