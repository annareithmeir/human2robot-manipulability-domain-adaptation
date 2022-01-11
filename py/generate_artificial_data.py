import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pandas as pd
from numpy import genfromtxt
from get_cov_ellipsoid import get_cov_ellipsoid, scale_volume, get_volume, scale_axes
from mpl_toolkits.mplot3d import Axes3D

colors=['green', 'blue', 'orange', 'red', 'purple']



# scaling parameters for toy data
volume_scaling = 2
axes_scaling = [2,0.5,1]

# Generate toy data manipulabilities 5000 random from panda random manipulabilities -> CALL generateRobotDataToy.m afterwards to create all necessary data

filename_manip = "/home/nnrthmr/PycharmProjects/ma_thesis/5000/data/panda/r_manipulabilities.csv"
filename_manip_toy = "/home/nnrthmr/PycharmProjects/ma_thesis/5000/data/toy_data/r_manipulabilities.csv"

manip_tmp = genfromtxt(filename_manip, delimiter=',')
manip_artificial_array=np.zeros((manip_tmp.shape[0], 9))

for i in np.arange(0, manip_tmp.shape[0]):
    m_i=manip_tmp[i,:].reshape(3,3)
    m_a = scale_volume(m_i, volume_scaling)
    m_a = scale_axes(m_a, axes_scaling)
    manip_artificial_array[i,:] = m_a.reshape(1,9)

np.savetxt(filename_manip_toy, manip_artificial_array, delimiter=",")



# Generate toy data from the reach up manipulabilities from panda for testing the mapping

fig = plt.figure()
ax = plt.axes(projection='3d')
plt.title('Demonstrations')



filename_manip = "/home/nnrthmr/CLionProjects/ma_thesis/data/demos/towards_singularities/panda/panda_reach_up_manipulabilities_interpolated.csv"
filename_manip_a = "/home/nnrthmr/CLionProjects/ma_thesis/data/demos/towards_singularities/panda/toy_data/panda_reach_up_manipulabilities_interpolated.csv"
manip_tmp = genfromtxt(filename_manip, delimiter=',')
manip_tmp=manip_tmp[1:,:]

n_points=manip_tmp.shape[0]
scaling_factor_plot = 0.5
plot_every_nth = 1

COLS=['s','x','y','z']

manip=list()
manip_artificial=list()
manip_artificial_array=np.zeros((manip_tmp.shape[0], 10))
manip_artificial_array[:,0]= manip_tmp[:,0]



for i in np.arange(0, manip_tmp.shape[0]):
    m_i=manip_tmp[i,1:].reshape(3,3)
    manip.append(scaling_factor_plot*m_i)

    m_a = scale_volume(m_i, volume_scaling)
    m_a = scale_axes(m_a, axes_scaling)
    manip_artificial.append(scaling_factor_plot*m_a)
    manip_artificial_array[i,1:] = m_a.reshape(1,9)


np.savetxt(filename_manip_a, manip_artificial_array, delimiter=",")

# cnt=0
# for i in np.arange(0,len(manip),plot_every_nth):
#     m_i = manip[i]
#     m_a = manip_artificial[i]

#     X2,Y2,Z2 = get_cov_ellipsoid(m_i, [1*cnt,0,0], 1)
#     ax.plot_wireframe(X2,Y2,Z2, color='blue', alpha=0.05)

#     X2,Y2,Z2 = get_cov_ellipsoid(m_a, [1*cnt,0,0], 1)
#     ax.plot_wireframe(X2,Y2,Z2, color='red', alpha=0.05)
#     cnt+=1



# # ax.set_zlim(-1, 1)
# # plt.xlim(-1 ,1)
# # plt.ylim(-1, 1)
# # plt.show()



# scale=np.diag([cnt, 1, 1, 1.0])
# scale=scale*(1.0/scale.max())
# scale[3,3]=0.7
# def short_proj():
#   return np.dot(Axes3D.get_proj(ax), scale)


# ax.get_proj=short_proj
# ax.set_box_aspect(aspect = (1,1,1))

# plt.xlim(-0.5, n_points/plot_every_nth)
# plt.ylim(-0.5, 0.5)
# ax.set_zlim(-0.5, 0.5)
# #plt.ylim(-0.5, n_points/plot_every_nth)
# #ax.set_zlim(-0.5, n_points/plot_every_nth)
# plt.show()


