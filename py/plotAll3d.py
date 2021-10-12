import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
from numpy import genfromtxt
import numpy as np
from scipy import linalg
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
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

n_demos=4
n_points=20
scaling_factor=0.01
plot_every_nth = 3

colors=['green', 'blue', 'orange', 'red', 'purple']

fig = plt.figure()
plt.subplot(1, 1, 1)
ax = plt.axes(projection='3d')
plt.title('Demonstrations and GMR results')

data_path = "../data/demos/trajectories.csv"
data = pd.read_csv(data_path, sep=",")
xdata= np.array(data['EE_x'])[:n_demos*n_points]
ydata= np.array(data['EE_y'])[:n_demos*n_points]
zdata= np.array(data['EE_z'])[:n_demos*n_points]
ax.scatter3D(xdata, ydata, zdata, c='grey', alpha=0.4)
ax.scatter3D(0,0,0, c='red', marker='x', s=100)

### plot demonstration manipulabilities ###

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

filename_mu = "../data/expData3d.csv"
filename_sigma= "../data/expDataSPD3d.csv"

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


import matplotlib.patches as mpatches

red_patch = mpatches.Patch(color='red', label='Robot base location')
blue_patch = mpatches.Patch(color='blue', label='Learned')
grey_patch = mpatches.Patch(color='grey', label='Demonstrated')

plt.legend(handles=[red_patch, blue_patch, grey_patch])
plt.xlim(-0.5, 0.5)
plt.ylim(-1, 1)
ax.set_zlim(0., 0.5)
plt.show()




