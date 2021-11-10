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







#filename_mu = sys.argv[1] # DxK, K Gaussians
#filename_sigma = sys.argv[2] #DxDxK

filename_mu = "../data/expData3d.csv"
filename_sigma= "../data/expDataSPD3d.csv"
#filename_sigma= "../data/expSigma3d.csv"
#filename_sigma= "/home/nnrthmr/Desktop/master-thesis/vrep/vrep_franka_promps/py_scripts/data/EEpos_translationmanipulability_trial_0.csv"

mu_tmp = genfromtxt(filename_mu, delimiter=',')
sigma_tmp = genfromtxt(filename_sigma, delimiter=',')

#df=pd.read_csv(filename_sigma, sep=",")
#df = df.drop(df.columns[[0]], axis=1)
#sigma_tmp=np.array(df)
sigma_tmp = 0.1*sigma_tmp

print(mu_tmp.shape)
print(sigma_tmp.shape)

mu=list()
sigma=list()

for i in np.arange(0, mu_tmp.shape[1],3):
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


plt.xlim(-1., 1.)
plt.ylim(-1., 1.)
plt.xlabel('m11')
plt.ylabel('m22')
#plt.xticks(())
#plt.yticks(())

plt.show()

if(len(sys.argv)==4):
    plt.save(sys.argv[3])



