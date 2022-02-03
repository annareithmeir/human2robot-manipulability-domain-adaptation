import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from pyriemann.utils.distance import distance_riemann, distance_wasserstein
import pandas as pd
from numpy import genfromtxt
import matplotlib.patches as mpatches
from get_cov_ellipsoid import get_cov_ellipsoid, scale_volume, get_volume, scale_axes, get_logeuclidean_distance
from mpl_toolkits.mplot3d import Axes3D
import argparse

colors=['green', 'blue', 'orange', 'red', 'purple']


### artificial dataset with volume and axes scaling ###
parser = argparse.ArgumentParser()
parser.add_argument("input_path", help="input_path.", type=str)
parser.add_argument("naive_path", help="naive_path.", type=str)
parser.add_argument("icp_path", help="icp_path", type=str)
parser.add_argument("groundtruth_path", help="groundtruth_path", type=str)

args = parser.parse_args()



fig = plt.figure()
ax = plt.axes(projection='3d')
plt.title('Mappings from Panda to Toy Data')

filename_manip_groundtruth = args.groundtruth_path
filename_manip_mapped_naive = args.naive_path
filename_manip_mapped_icp = args.icp_path
filename_manips_input = args.input_path

manip_groundtruth = genfromtxt(filename_manip_groundtruth, delimiter=',')
#manip_groundtruth=manip_groundtruth[1:,:]
#print(manip_groundtruth)
manip_naive = genfromtxt(filename_manip_mapped_naive, delimiter=',')
#manip_naive=manip_naive[1:,:]
manip_icp = genfromtxt(filename_manip_mapped_icp, delimiter=',')
manip_input = genfromtxt(filename_manips_input, delimiter=',')
# manip_input=manip_input[1:,:]

n_points=manip_groundtruth.shape[0]
scaling_factor_plot = 0.3
plot_every_nth = 1

COLS=['s','x','y','z']

cnt=0
mse_naive=0.0
mse_icp=0.0

le_mse_naive=0.0
le_mse_icp=0.0

print("Errors between groundtruth and naive/icp")
print(manip_groundtruth.shape)
print(manip_naive.shape)
print(manip_icp.shape)
print(manip_input.shape)
for i in np.arange(0,n_points,plot_every_nth):
    m_i = manip_groundtruth[i,1:].reshape(3,3)
    m_i_n = manip_naive[i,:].reshape(3,3)
    m_i_icp = manip_icp[i,:].reshape(3,3)
    m_i_in = manip_input[i,1:].reshape(3,3)

    print("%.3f / %.3f" %(distance_riemann(m_i, m_i_n), distance_riemann(m_i, m_i_icp)))
    mse_icp += distance_riemann(m_i, m_i_icp)**2
    mse_naive+= distance_riemann(m_i, m_i_n)**2

    le_mse_icp += get_logeuclidean_distance(m_i, m_i_icp)**2
    le_mse_naive+= get_logeuclidean_distance(m_i, m_i_n)**2

    m_i=scaling_factor_plot*m_i
    m_i_n=scaling_factor_plot*m_i_n
    m_i_in=scaling_factor_plot*m_i_in
    m_i_icp=scaling_factor_plot*m_i_icp

    X2,Y2,Z2 = get_cov_ellipsoid(m_i, [1*cnt,0,0], 1)
    ax.plot_wireframe(X2,Y2,Z2, color='blue', alpha=0.05)

    X2,Y2,Z2 = get_cov_ellipsoid(m_i_n, [1*cnt,0,0], 1)
    ax.plot_wireframe(X2,Y2,Z2, color='red', alpha=0.05)

    X2,Y2,Z2 = get_cov_ellipsoid(m_i_icp, [1*cnt,0,0], 1)
    ax.plot_wireframe(X2,Y2,Z2, color='orange', alpha=0.05)

    X2,Y2,Z2 = get_cov_ellipsoid(m_i_in, [1*cnt,0,0], 1)
    ax.plot_wireframe(X2,Y2,Z2, color='green', alpha=0.05)

    cnt+=1

print("MSE (riemann): %.3f / %.3f" %(mse_naive/n_points, mse_icp/n_points))
print("MSE (LogEuc): %.3f / %.3f" %(le_mse_naive/n_points, le_mse_icp/n_points))
plt.title("MSE (riemann): %.3f / %.3f" %(mse_naive/n_points, mse_icp/n_points))

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
orange_patch = mpatches.Patch(color='orange', label='ICP')
green_patch = mpatches.Patch(color='green', label='Input')
plt.legend(handles=[ blue_patch, red_patch, orange_patch, green_patch])

ax.get_proj=short_proj
ax.set_box_aspect(aspect = (1,1,1))

plt.xlim(-0.5, n_points/plot_every_nth)
plt.ylim(-0.5, 0.5)
ax.set_zlim(-0.5, 0.5)
#plt.ylim(-0.5, n_points/plot_every_nth)
#ax.set_zlim(-0.5, n_points/plot_every_nth)

results_path = "/".join(str.split(filename_manip_mapped_icp, "/")[:-1])
robot_teacher= str.split(filename_manips_input, "/")[-3]
robot_student= str.split(filename_manip_mapped_icp, "/")[-3]
plt.savefig(results_path+"/mapping_from_"+robot_teacher+"_to_"+robot_student+"_plot.pdf")
plt.show()

