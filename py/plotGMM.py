import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
from numpy import genfromtxt
import numpy as np
from scipy import linalg



#filename_mu = sys.argv[1] # DxK, K Gaussians
#filename_sigma = sys.argv[2] #DxDxK

#mu = genfromtxt(filename_mu, delimiter=',')
#sigma = genfromtxt(filename_sigma, delimiter=',')

mu_d=np.array([0,0])
sigma_d=np.array([[304.380134285343,	-188.288919971871],[-188.288919971871,	124.941294748085]])


mu=[np.array([0,0]), np.array([10,10])]
sigma=[np.array([[49.8564064605510,	1.07179676972449],[1.07179676972449,	30.1435935394490]]),
np.array([[49.8564064605510,	1.07179676972449],[1.07179676972449,	30.1435935394490]])]

print(len(mu))
nstates = len(mu)
print('nstates: ', nstates)

splot = plt.subplot(1, 1, 1)
# plot desired
v, w = linalg.eigh(sigma_d)
v = 2. * np.sqrt(2.) * np.sqrt(v)
u = w[0] / linalg.norm(w[0])

angle = np.arctan(u[1] / u[0])
angle = 180. * angle / np.pi  # convert to degrees
ell = mpl.patches.Ellipse(mu_d, v[0], v[1], 180. + angle, color='navy', ls='--')
#ell.set_clip_box(splot.bbox)
ell.set_alpha(0.5)
splot.add_artist(ell)


# plot current
for i in np.arange(nstates):
    mu_i = mu[i]
    sigma_i=sigma[i]
    v, w = linalg.eigh(sigma_i)
    v = 2. * np.sqrt(2.) * np.sqrt(v)
    u = w[0] / linalg.norm(w[0])

    angle = np.arctan(u[1] / u[0])
    angle = 180. * angle / np.pi  # convert to degrees
    ell = mpl.patches.Ellipse(mu_i, v[0], v[1], 180. + angle, color='gold')
    #ell.set_clip_box(splot.bbox)
    ell.set_alpha(0.5)
    splot.add_artist(ell)

plt.xlim(-50., 50.)
plt.ylim(-50., 50.)
plt.xlabel('m11')
plt.ylabel('m22')
#plt.xticks(())
#plt.yticks(())

plt.show()

if(len(sys.argv)==4):
    plt.save(sys.argv[3])



