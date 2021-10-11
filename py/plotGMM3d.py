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





#filename_mu = sys.argv[1] # DxK, K Gaussians
#filename_sigma = sys.argv[2] #DxDxK

filename_mu = "../data/expData3d.csv"
#filename_sigma= "../data/expDataSPD3d.csv"
filename_sigma= "../data/expSigma3d.csv"
#filename_sigma= "/home/nnrthmr/Desktop/master-thesis/vrep/vrep_franka_promps/py_scripts/data/EEpos_translationmanipulability_trial_0.csv"

#filename_mu = sys.argv[1] # DxK, K Gaussians
#filename_sigma = sys.argv[2] #Kx(D*D)

mu_tmp = genfromtxt(filename_mu, delimiter=',')
#sigma_tmp = genfromtxt(filename_sigma, delimiter=',')

df=pd.read_csv(filename_sigma, sep=",")
#df = df.drop(df.columns[[0]], axis=1)
sigma_tmp=np.array(df)
sigma_tmp = 1.0*sigma_tmp

print(mu_tmp.shape)
print(sigma_tmp.shape)

mu=list()
sigma=list()

for i in np.arange(0, mu_tmp.shape[1], 5):
    mu.append(mu_tmp[:,i])
    sigma.append(sigma_tmp[i,:].reshape(3,3))

nstates = len(mu)
print('nstates: ', nstates)

splot = plt.subplot(1, 1, 1)
ax = plt.axes(projection='3d')

# plot current
for i in np.arange(nstates):
    mu_i = mu[i]
    sigma_i=sigma[i]
    X2,Y2,Z2 = get_cov_ellipsoid(sigma_i, mu_i, 3)
    ax.plot_wireframe(X2,Y2,Z2, color='b', alpha=0.1)


plt.xlim(-15., 15.)
plt.ylim(-15., 15.)
plt.xlabel('m11')
plt.ylabel('m22')
#plt.xticks(())
#plt.yticks(())

plt.show()

if(len(sys.argv)==4):
    plt.save(sys.argv[3])



