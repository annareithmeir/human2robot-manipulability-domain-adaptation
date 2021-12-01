import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
from numpy import genfromtxt
import numpy as np
from scipy import linalg
from mpl_toolkits import mplot3d
from numpy import linalg as LA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import pandas as pd
from get_cov_ellipsoid import get_cov_ellipsoid


colors=['green', 'blue', 'orange', 'red', 'purple']



one_loop = False


###########################
### plot demonstrations ###

n_demos=1
n_points=100
scaling_factor=1
plot_every_nth = 1

#REACH_UP=False # copied manip from reaching up task
#if REACH_UP:
#    tmp = np.array([0.112386331709752,	-0.292084314237333,	0.085136827901769,	-0.292084314237333,	0.892749865686052,	0.007596007158765,	0.085136827901769,	0.007596007158765,	0.698160049993724])



fig = plt.figure()
ax = plt.axes(projection='3d')
plt.title('desired and mapped manipulabilities')
'''
if one_loop:

    ##############################################
    #   vrep simple data first control loop      #
    ##############################################
    filename_err= ""
    #filename_demos = "../data/demos/translationManip3d.csv"
    #filename_controlled= "../data/tracking/loopManipulabilities.csv"
    filename_controlled="/home/nnrthmr/Desktop/manips1.csv"

    n_points = 100
    plot_every_nth = 5
    n_demos = 1



##############################################
#   vrep simple data                         #
##############################################
filename_err= "../data/tracking/errorManipulabilities.csv"
filename_demos = "../data/demos/translationManip3d.csv"
filename_controlled= "../data/tracking/xhat.csv"


##############################################
#   human arm data                           #
##############################################
filename_demos = "../data/demos/humanArm/dummyManipulabilities.csv"
filename_controlled= "../data/results/human_arm/xhat.csv"
'''

### plot desired manipulabilities ###
scaling_factor = 0.1
n_points=1000
plot_every_nth=250
filename_demos="/home/nnrthmr/CLionProjects/ma_thesis/data/learning/rhuman/cut_userchoice/4/xhat.csv"
#filename_demos="/home/nnrthmr/CLionProjects/ma_thesis/data/learning/rhuman/cut_random/3/xhat.csv"
#filename_demos="/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/affineTrafo/h_manipulabilities_normalized.csv"
#filename_demos="/home/nnrthmr/Desktop/someData.csv"
demo_tmp = genfromtxt(filename_demos, delimiter=',')
#demo_tmp=demo_tmp[:10,:]
demo=list()

if one_loop:
    for i in np.arange(0, n_points):
        if REACH_UP:
            demo.append(scaling_factor*tmp.reshape(3,3))
        else:
            demo.append(scaling_factor*demo_tmp[0,1:].reshape(3,3))
else:
    for i in np.arange(0, demo_tmp.shape[0]):
        demo.append(scaling_factor*demo_tmp[i,0:].reshape(3,3))


demo=demo[:n_demos*n_points]

cnt=0
for i in np.arange(0,len(demo),plot_every_nth):
    m_i = demo[i]


    X2,Y2,Z2 = get_cov_ellipsoid(m_i, [1*cnt,0,0], 1)
    ax.plot_wireframe(X2,Y2,Z2, color='red', alpha=0.05)
    cnt+=1



### plot mapped manipulabilities ###
#filename_controlled="/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/cut_random/3/mapped_manipulabilities.csv"
#filename_controlled="/home/nnrthmr/Desktop/someDataMapped_log_exp.csv"
#filename_controlled="/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/cut_random/3/mapped_manipulabilities_log_exp.csv"
filename_controlled="/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/cut_userchoice/4/mapped_manipulabilities_log_exp.csv"
controlled_tmp = genfromtxt(filename_controlled, delimiter=',')
controlled_tmp = scaling_factor*controlled_tmp

controlled=list()
for i in np.arange(n_points):
    controlled.append(controlled_tmp[i,0:].reshape(3,3))

cnt=0
for i in np.arange(0,n_points,plot_every_nth):
    controlled_i=controlled[i]
    X2,Y2,Z2 = get_cov_ellipsoid(controlled_i, [1*cnt,0,0], 1)
    ax.plot_wireframe(X2,Y2,Z2, color='blue', alpha=0.1)
    cnt+=1

# test=np.array([     0.3231 ,   0.0010 ,   0.1556,
#     0.0010 ,   0.4291  , -0.1426,
#     0.1556 ,  -0.1426  ,  0.5337]).reshape(3,3) * scaling_factor

# X2,Y2,Z2 = get_cov_ellipsoid(test, [0,0,0], 1)
# ax.plot_wireframe(X2,Y2,Z2, color='black', alpha=0.1)


blue_patch = mpatches.Patch(color='blue', label='Mapped')
#grey_patch = mpatches.Patch(color='gold', label='Demonstrated')
red_patch = mpatches.Patch(color='red', label='Desired')

scale=np.diag([cnt, 1, 1, 1.0])
scale=scale*(1.0/scale.max())
scale[3,3]=0.7
def short_proj():
  return np.dot(Axes3D.get_proj(ax), scale)


ax.get_proj=short_proj
ax.set_box_aspect(aspect = (1,1,1))

plt.legend(handles=[ blue_patch, red_patch])
plt.xlim(-0.5, n_points/plot_every_nth)
plt.ylim(-0.5, 0.5)
ax.set_zlim(-0.5, 0.5)
#plt.ylim(-0.5, n_points/plot_every_nth)
#ax.set_zlim(-0.5, n_points/plot_every_nth)
plt.show()






