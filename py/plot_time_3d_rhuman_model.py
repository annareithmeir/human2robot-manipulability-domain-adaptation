import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
from numpy import genfromtxt
import numpy as np
from scipy import linalg
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import pandas as pd
from get_cov_ellipsoid import get_cov_ellipsoid


colors=['green', 'blue', 'orange', 'red', 'purple']



one_loop = False
plot_demonstrations = False
plot_controlled = False
plot_error = False


###########################
### plot demonstrations ###

n_demos=1
scaling_factor=1e-1
plot_every_nth = 30

fig = plt.figure()
plt.subplot(1, 2, 1)
ax = plt.axes(projection='3d')
plt.title('Maniulabilities over time')

if one_loop:

    ##############################################
    #   vrep simple data first control loop      #
    ##############################################
    plot_error = False
    filename_demos = "../data/demos/translationManip3d.csv"
    filename_controlled= "../data/tracking/loopManipulabilities.csv"

    n_points = 3000
    plot_every_nth = 1000
    n_demos = 1


'''
##############################################
#   vrep simple data                         #
##############################################
filename_err= "../data/tracking/errorManipulabilities.csv"
filename_demos = "../data/demos/translationManip3d.csv"
filename_controlled= "../data/tracking/xhat.csv"
'''


### plot demonstration manipulabilities ###
if plot_demonstrations:
    filename_demos = "/home/nnrthmr/Desktop/RHuMAn-arm-model/data/drill_optimal/exp2_task19_m.csv"
    demo_tmp = genfromtxt(filename_demos, delimiter=',')
    demo_tmp=demo_tmp[:,:]
    demo=list()

    if one_loop:
        for i in np.arange(0, demo_tmp.shape[0]):
            demo.append(scaling_factor*demo_tmp[0,1:].reshape(3,3))
    else:
        for i in np.arange(0, demo_tmp.shape[0]):
            demo.append(scaling_factor*demo_tmp[i,1:].reshape(3,3))


    cnt=0
    for i in np.arange(0,len(demo),plot_every_nth):
        m_i = demo[i]
        X2,Y2,Z2 = get_cov_ellipsoid(m_i, [2*cnt,0,0], 1)
        ax.plot_wireframe(X2,Y2,Z2, color='orange', alpha=0.05)
        cnt+=1


### plot controlled manipulabilities ###
if plot_controlled:
    filename_controlled= "../data/results/human_arm/xhat.csv"
    controlled_tmp = genfromtxt(filename_controlled, delimiter=',')
    controlled_tmp = scaling_factor*controlled_tmp

    controlled=list()
    for i in np.arange(demo_tmp.shape[0]):
        controlled.append(controlled_tmp[i,:].reshape(3,3))

    cnt=0
    for i in np.arange(0,demo_tmp.shape[0],plot_every_nth):
        controlled_i=controlled[i]
        X2,Y2,Z2 = get_cov_ellipsoid(controlled_i, [2*cnt,0,0], 1)
        ax.plot_wireframe(X2,Y2,Z2, color='blue', alpha=0.1)
        cnt+=1



### plot error ###
if(plot_error):
    err_tmp = genfromtxt(filename_err, delimiter=',')

    err=list()
    for i in np.arange(demo_tmp.shape[0]):
        err.append(err_tmp[i])

    x=np.arange(int(demo_tmp.shape[0]/plot_every_nth))
    y=np.ones(int(demo_tmp.shape[0]/plot_every_nth))
    ax.plot(x,y,np.array(err)-np.mean(err), color='red', alpha=0.5)


### General stuff ###

scale=np.diag([4, 1, 1, 1.0])
scale=scale*(1.0/scale.max())
scale[3,3]=1.0

def short_proj():
  return np.dot(Axes3D.get_proj(ax), scale)

ax.get_proj=short_proj

blue_patch = mpatches.Patch(color='blue', label='Controlled')
grey_patch = mpatches.Patch(color='orange', label='Demonstrated')
red_patch = mpatches.Patch(color='red', label='Error')

plt.legend(handles=[ blue_patch, grey_patch, red_patch])
plt.xlim(-0.5, 2*demo_tmp.shape[0]/plot_every_nth)
plt.ylim(-1, 1)
ax.set_zlim(-1, 1)
plt.show()






