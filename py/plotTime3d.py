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

colors=['green', 'blue', 'orange', 'red', 'purple']

def get_cov_ellipsoid(cov, mu=np.zeros((3)), nstd=3):
    """
    Return the 3d points representing the covariance matrix
    cov centred at mu and scaled by the factor nstd.
    Plot on your favourite 3d axis. 
    Example 1:  ax.plot_wireframe(X,Y,Z,alpha=0.1)
    Example 2:  ax.plot_surface(X,Y,Z,alpha=0.1)
    """
    assert cov.shape==(3,3)

    # Find and sort eigenvalues to correspond to the covariance matrix
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.sum(cov,axis=0).argsort()
    eigvals_temp = eigvals[idx]
    idx = eigvals_temp.argsort()
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:,idx]

    # Set of all spherical angles to draw our ellipsoid
    n_points = 100
    theta = np.linspace(0, 2*np.pi, n_points)
    phi = np.linspace(0, np.pi, n_points)

    # Width, height and depth of ellipsoid
    rx, ry, rz = nstd * np.sqrt(eigvals)

    # Get the xyz points for plotting
    # Cartesian coordinates that correspond to the spherical angles:
    X = rx * np.outer(np.cos(theta), np.sin(phi))
    Y = ry * np.outer(np.sin(theta), np.sin(phi))
    Z = rz * np.outer(np.ones_like(theta), np.cos(phi))

    # Rotate ellipsoid for off axis alignment
    old_shape = X.shape
    # Flatten to vectorise rotation
    X,Y,Z = X.flatten(), Y.flatten(), Z.flatten()
    X,Y,Z = np.matmul(eigvecs, np.array([X,Y,Z]))
    X,Y,Z = X.reshape(old_shape), Y.reshape(old_shape), Z.reshape(old_shape)
   
    # Add in offsets for the mean
    X = X + mu[0]
    Y = Y + mu[1]
    Z = Z + mu[2]
    
    return X,Y,Z


###########################
### plot demonstrations ###

n_demos=1
n_points=20
#n_points=400
scaling_factor=1e-1
plot_every_nth = 1

colors=['green', 'blue', 'orange', 'red', 'purple']

fig = plt.figure()
plt.subplot(1, 2, 1)
ax = plt.axes(projection='3d')
plt.title('Demonstrations and Control results')


### plot demonstration manipulabilities ###
filename_manip = "../data/demos/translationManip3d.csv"
#filename_manip = "../data/demos/humanArm/dummyManipulabilities.csv"
manip_tmp = genfromtxt(filename_manip, delimiter=',')
manip_tmp=manip_tmp[1:,:]
manip=list()

for i in np.arange(0, manip_tmp.shape[0]):
    manip.append(scaling_factor*manip_tmp[i,1:].reshape(3,3))

manip=manip[:n_demos*n_points]

for i in np.arange(0,len(manip),plot_every_nth):
    m_i = manip[i]
    X2,Y2,Z2 = get_cov_ellipsoid(m_i, [1*i,0,0], 1)
    ax.plot_wireframe(X2,Y2,Z2, color='gold', alpha=0.05)


### plot controlled manipulabilities ###
filename_sigma= "../data/tracking/xhat.csv"
#filename_sigma= "../data/results/human_arm/xhat.csv"
sigma_tmp = genfromtxt(filename_sigma, delimiter=',')
sigma_tmp = scaling_factor*sigma_tmp

sigma=list()
for i in np.arange(n_points):
    sigma.append(sigma_tmp[i,:].reshape(3,3))

for i in np.arange(0,n_points,plot_every_nth):
    sigma_i=sigma[i]
    X2,Y2,Z2 = get_cov_ellipsoid(sigma_i, [1*i,0,0], 1)
    ax.plot_wireframe(X2,Y2,Z2, color='blue', alpha=0.1)



### plot error ###
filename_err= "../data/tracking/errorManipulabilities.csv"
err_tmp = genfromtxt(filename_err, delimiter=',')

err=list()
for i in np.arange(n_points):
    err.append(err_tmp[i])

#err=np.arange(-0.5, -0.5+0.01*int(n_points/plot_every_nth), 0.01)

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






