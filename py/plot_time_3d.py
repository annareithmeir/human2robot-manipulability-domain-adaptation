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
import get_cov_ellipsoid


colors=['green', 'blue', 'orange', 'red', 'purple']



one_loop = True


###########################
### plot demonstrations ###

n_demos=1
n_points=20
#n_points=400
scaling_factor=1e-1
plot_every_nth = 1

REACH_UP=False # copied manip from reaching up task
if REACH_UP:
    tmp = np.array([0.112386331709752,	-0.292084314237333,	0.085136827901769,	-0.292084314237333,	0.892749865686052,	0.007596007158765,	0.085136827901769,	0.007596007158765,	0.698160049993724])

colors=['green', 'blue', 'orange', 'red', 'purple']

fig = plt.figure()
plt.subplot(1, 2, 1)
ax = plt.axes(projection='3d')
#plt.title('Demonstrations and Control results)
plt.title('Control loop - Km=adaptive, n_iter=3000, final_err=0.97')

if one_loop:

    ##############################################
    #   vrep simple data first control loop      #
    ##############################################
    filename_err= ""
    filename_demos = "../data/demos/translationManip3d.csv"
    filename_controlled= "../data/tracking/loopManipulabilities.csv"

    n_points = 3000
    plot_every_nth = 100
    n_demos = 1


'''
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
filename_controlled= "../data/results/human_arm/xhat.csv
'''

### plot demonstration manipulabilities ###
demo_tmp = genfromtxt(filename_demos, delimiter=',')
demo_tmp=demo_tmp[1:,:]
demo=list()

if one_loop:
    for i in np.arange(0, n_points):
        if REACH_UP:
            demo.append(scaling_factor*tmp.reshape(3,3))
        else:
            demo.append(scaling_factor*demo_tmp[0,1:].reshape(3,3))
else:
    for i in np.arange(0, demo_tmp.shape[0]):
        demo.append(scaling_factor*demo_tmp[i,1:].reshape(3,3))


demo=demo[:n_demos*n_points]

cnt=0
for i in np.arange(0,len(demo),plot_every_nth):
    m_i = demo[i]
    X2,Y2,Z2 = get_cov_ellipsoid(m_i, [1*cnt,0,0], 1)
    ax.plot_wireframe(X2,Y2,Z2, color='gold', alpha=0.05)
    cnt+=1


### plot controlled manipulabilities ###
controlled_tmp = genfromtxt(filename_controlled, delimiter=',')
controlled_tmp = scaling_factor*controlled_tmp

controlled=list()
for i in np.arange(n_points):
    controlled.append(controlled_tmp[i,:].reshape(3,3))

cnt=0
for i in np.arange(0,n_points,plot_every_nth):
    controlled_i=controlled[i]
    X2,Y2,Z2 = get_cov_ellipsoid(controlled_i, [1*cnt,0,0], 1)
    ax.plot_wireframe(X2,Y2,Z2, color='blue', alpha=0.1)
    cnt+=1



### plot error ###
if(filename_err!=""):
    err_tmp = genfromtxt(filename_err, delimiter=',')

    err=list()
    for i in np.arange(n_points):
        err.append(err_tmp[i])

    x=np.arange(int(n_points/plot_every_nth))
    y=np.ones(int(n_points/plot_every_nth))
    ax.plot(x,y,np.array(err)-np.mean(err), color='red', alpha=0.5)

blue_patch = mpatches.Patch(color='blue', label='Controlled')
grey_patch = mpatches.Patch(color='gold', label='Demonstrated')
red_patch = mpatches.Patch(color='red', label='Error')

plt.legend(handles=[ blue_patch, grey_patch, red_patch])
plt.xlim(-0.5, n_points/plot_every_nth)
plt.ylim(-1, 1)
ax.set_zlim(-1, 1)
plt.show()






