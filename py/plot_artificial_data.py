import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pandas as pd
from numpy import genfromtxt
import matplotlib.patches as mpatches
from get_cov_ellipsoid import get_cov_ellipsoid, scale_volume, get_volume, scale_axes, get_logeuclidean_distance
from mpl_toolkits.mplot3d import Axes3D

colors=['green', 'blue', 'orange', 'red', 'purple']


# Generate toy data from the reach up manipulabilities from panda for testing the mapping

fig = plt.figure()
ax = plt.axes(projection='3d')
plt.title('Mappings from Panda to Toy Data')



filename_manip_true = "/home/nnrthmr/CLionProjects/ma_thesis/data/demos/towards_singularities/data/panda/toy_data/panda_reach_up_manipulabilities_interpolated.csv"
filename_manip_mapped_naive = "/home/nnrthmr/PycharmProjects/ma_thesis/5000/results/toy_data/mapped_manipulabilities_human_naive.csv"
filename_manips_input = "/home/nnrthmr/CLionProjects/ma_thesis/data/demos/towards_singularities/data/panda/panda_reach_up_manipulabilities_interpolated.csv"
manip_true = genfromtxt(filename_manip_true, delimiter=',')
manip_true=manip_true[1:,:]
manip_naive = genfromtxt(filename_manip_mapped_naive, delimiter=',')
manip_naive=manip_naive[1:,:]
manip_input = genfromtxt(filename_manips_input, delimiter=',')
manip_input=manip_input[1:,:]

n_points=manip_true.shape[0]
scaling_factor_plot = 0.5
plot_every_nth = 1

COLS=['s','x','y','z']

manip=list()
manip_n=list()
manip_in=list()


for i in np.arange(0, manip_true.shape[0]):
    m_i=manip_true[i,1:].reshape(3,3)
    manip.append(scaling_factor_plot*m_i)

    m_naive=manip_naive[i,:].reshape(3,3)
    manip_n.append(scaling_factor_plot*m_naive)

    m_in=manip_input[i,1:].reshape(3,3)
    manip_in.append(scaling_factor_plot*m_in)


cnt=0
for i in np.arange(0,len(manip),plot_every_nth):
    m_i = manip[i]
    m_i_n = manip_n[i]
    m_i_in = manip_in[i]

    X2,Y2,Z2 = get_cov_ellipsoid(m_i, [1*cnt,0,0], 1)
    ax.plot_wireframe(X2,Y2,Z2, color='blue', alpha=0.05)

    X2,Y2,Z2 = get_cov_ellipsoid(m_i_n, [1*cnt,0,0], 1)
    ax.plot_wireframe(X2,Y2,Z2, color='red', alpha=0.05)

    X2,Y2,Z2 = get_cov_ellipsoid(m_i_in, [1*cnt,0,0], 1)
    ax.plot_wireframe(X2,Y2,Z2, color='green', alpha=0.05)

    print(get_logeuclidean_distance(m_i, m_i_n))
    cnt+=1



# ax.set_zlim(-1, 1)
# plt.xlim(-1 ,1)
# plt.ylim(-1, 1)
# plt.show()



scale=np.diag([cnt, 1, 1, 1.0])
scale=scale*(1.0/scale.max())
scale[3,3]=0.7
def short_proj():
  return np.dot(Axes3D.get_proj(ax), scale)


blue_patch = mpatches.Patch(color='blue', label='Ground Truth')
red_patch = mpatches.Patch(color='red', label='Naive')
green_patch = mpatches.Patch(color='green', label='Input')
plt.legend(handles=[ blue_patch, red_patch, green_patch])

ax.get_proj=short_proj
ax.set_box_aspect(aspect = (1,1,1))

plt.xlim(-0.5, n_points/plot_every_nth)
plt.ylim(-0.5, 0.5)
ax.set_zlim(-0.5, 0.5)
#plt.ylim(-0.5, n_points/plot_every_nth)
#ax.set_zlim(-0.5, n_points/plot_every_nth)
plt.show()

