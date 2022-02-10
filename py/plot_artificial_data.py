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
from vectorizeSPD import SPD_from_8d


def get_errors(data1p, data2p):

  data1 = genfromtxt(data1p, delimiter=',')
  data2 = genfromtxt(data2p, delimiter=',')
  print(data1.shape)
  print(data2.shape)

  if "8d" in data2p: # 8d data from cpd
     data2 = SPD_from_8d(data2) #list

  if data1.shape[1]==10:
    data1=data1[:,1:]

  mse_icp=0.0
  le_mse_icp=0.0

  for (mi,mj) in zip(data1, data2):
    mi=mi.reshape(3,3)
    mj=mj.reshape(3,3)
    print("%.3f " %(distance_riemann(mi, mj)))
    mse_icp += distance_riemann(mi, mj)**2
    le_mse_icp += get_logeuclidean_distance(mi, mj)**2

  n_points = data1.shape[0]
  print("MSE (riemann): %.3f " %(mse_icp/n_points))
  print("MSE (LogEuc): %.3f " %(le_mse_icp/n_points))




### artificial dataset with volume and axes scaling ###
parser = argparse.ArgumentParser()
#parser.add_argument("input_path", help="input_path.", type=str)
#parser.add_argument("groundtruth_path", help="groundtruth_path", type=str)
parser.add_argument('-mapping_paths','--l', nargs='+', type=str)
#parser.add_argument("icp_path", help="icp_path", type=str)

args = parser.parse_args()
args.paths = [item for item in args.l[0].split(',')]


colors=['cornflowerblue', 'darkblue', 'mediumorchid', 'plum', 'darkorange']
# labellist=['input (RHuman)','CPD','mapped2']
# labellist=['input (RHuman)','ICP','CPD','mapped2']
labellist=['$input$','$ground truth$','$ICP$']
scaling_factor=1e-1
plot_every_nth=1



### front view ###
fig = plt.figure()
fig.subplots_adjust(bottom=-0.15,top=1.2,wspace=0, hspace=0, right=1)

#ax = fig.add_subplot(2,1,1, projection='3d')
#ax2 = fig.add_subplot(2,1,2, projection='3d')
plt.suptitle('Mappings from Panda to Toy Data')
ax = plt.axes(projection='3d')

c=0
mse_icp=0.0

le_mse_naive=0.0
le_mse_icp=0.0

for p in args.paths:
  data = genfromtxt(p, delimiter=',')

  if "8d" in p: # 8d data from cpd
       data = SPD_from_8d(data) #list

  if data.shape[1]==10:
    data=data[:,1:]

  manip=list()

  for i in np.arange(data.shape[0]):
    mm=data[i].reshape(3,3)
    manip.append(scaling_factor*mm)

  cnt=0
  for i in np.arange(0,len(manip),plot_every_nth):
    m_i = manip[i]
    w,v = np.linalg.eigh(m_i) # just for very singular cases as in trajectories generated
    w[w<1e-12]=0.01
    m=np.matmul(np.matmul(v, np.diag(w)), v.transpose())
    m_i=m
    X2,Y2,Z2 = get_cov_ellipsoid(m_i, [0.5*cnt,0,0], 1)
    ax.plot_wireframe(X2,Y2,Z2, color=colors[c], alpha=0.05)
    cnt+=1
  c+=1


# print errors
#get_errors(args.paths[0],args.paths[1])
get_errors(args.paths[1],args.paths[2])

scale=np.diag([0.5*cnt, 1, 1, 1.0])
scale=scale*(1.0/scale.max())
scale[3,3]=0.7
def short_proj():
  return np.dot(Axes3D.get_proj(ax), scale)

ax.get_proj=short_proj
#ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([cnt, 1, 1, 1]))
ax.set_box_aspect(aspect = (1,1,1))
ax.view_init(0, 90)


blue_patch = mpatches.Patch(color='cornflowerblue', label=labellist[0])
red_patch = mpatches.Patch(color='darkblue', label=labellist[1])
if len(args.paths)==3:
  orange_patch = mpatches.Patch(color='mediumorchid', label=labellist[2])
  ax.legend(handles=[ blue_patch, red_patch, orange_patch], loc='center left', bbox_to_anchor=(1.07, 0.5))
elif len(args.paths)==4:
  orange_patch = mpatches.Patch(color='plum', label=labellist[3])
  ax.legend(handles=[ blue_patch, red_patch, orange_patch], loc='center left', bbox_to_anchor=(1.07, 0.5))
else:
  ax.legend(handles=[ blue_patch, red_patch], loc='center left', bbox_to_anchor=(1.07, 0.5))

plt.xlim(-0.5, 0.5*len(manip)/plot_every_nth)
plt.ylim(-0.5, 0.5)
ax.set_zlim(-0.5, 0.5)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')

ax.set_title("Front view")

plt.tight_layout()
#plt.show()



### top view ###
fig2 = plt.figure()
fig2.subplots_adjust(bottom=-0.15,top=1.2,wspace=0, hspace=0, right=1)

plt.suptitle('Mappings from Panda to Toy Data')
ax2 = plt.axes(projection='3d')

c=0
mse_icp=0.0

le_mse_naive=0.0
le_mse_icp=0.0

for p in args.paths:
  data = genfromtxt(p, delimiter=',')

  if "8d" in p: # 8d data from cpd
       data = SPD_from_8d(data) #list

  if data.shape[1]==10:
    data=data[:,1:]

  manip=list()

  for i in np.arange(data.shape[0]):
    mm=data[i].reshape(3,3)
    manip.append(scaling_factor*mm)

  cnt=0
  for i in np.arange(0,len(manip),plot_every_nth):
    m_i = manip[i]
    w,v = np.linalg.eigh(m_i) # just for very singular cases as in trajectories generated
    w[w<1e-12]=0.01
    m=np.matmul(np.matmul(v, np.diag(w)), v.transpose())
    m_i=m
    X2,Y2,Z2 = get_cov_ellipsoid(m_i, [0.5*cnt,0,0], 1)
    ax2.plot_wireframe(X2,Y2,Z2, color=colors[c], alpha=0.05)
    cnt+=1
  c+=1


scale=np.diag([0.5*cnt, 1, 1, 1.0])
scale=scale*(1.0/scale.max())
scale[3,3]=0.7
def short_proj2():
  return np.dot(Axes3D.get_proj(ax2), scale)

ax2.get_proj=short_proj2
#ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([cnt, 1, 1, 1]))
ax2.set_box_aspect(aspect = (1,1,1))
ax2.view_init(azim=90, elev=90)


blue_patch = mpatches.Patch(color='cornflowerblue', label=labellist[0])
red_patch = mpatches.Patch(color='darkblue', label=labellist[1])
if len(args.paths)==3:
  orange_patch = mpatches.Patch(color='mediumorchid', label=labellist[2])
  ax2.legend(handles=[ blue_patch, red_patch, orange_patch], loc='center left', bbox_to_anchor=(1.07, 0.5))
elif len(args.paths)==4:
  orange_patch = mpatches.Patch(color='plum', label=labellist[3])
  ax2.legend(handles=[ blue_patch, red_patch, orange_patch], loc='center left', bbox_to_anchor=(1.07, 0.5))
else:
  ax2.legend(handles=[ blue_patch, red_patch], loc='center left', bbox_to_anchor=(1.07, 0.5))

plt.xlim(-0.5, 0.5*len(manip)/plot_every_nth)
plt.ylim(-0.5, 0.5)
ax2.set_zlim(-0.5, 0.5)
ax2.set_xlabel('$x$')
ax2.set_ylabel('$y$')
ax2.set_zlabel('$z$')

ax2.set_title("Top view")

plt.tight_layout()
plt.show()










# fig = plt.figure()
# ax = plt.axes(projection='3d')
# plt.title('Mappings from Panda to Toy Data')

# filename_manip_groundtruth = args.groundtruth_path
# filename_manip_mapped_naive = args.naive_path
# filename_manip_mapped_icp = args.icp_path
# filename_manips_input = args.input_path

# manip_groundtruth = genfromtxt(filename_manip_groundtruth, delimiter=',')
# #manip_groundtruth=manip_groundtruth[1:,:]
# #print(manip_groundtruth)
# manip_naive = genfromtxt(filename_manip_mapped_naive, delimiter=',')
# #manip_naive=manip_naive[1:,:]
# manip_icp = genfromtxt(filename_manip_mapped_icp, delimiter=',')
# manip_input = genfromtxt(filename_manips_input, delimiter=',')
# # manip_input=manip_input[1:,:]

# if "8d" in filename_manip_mapped_icp: # 8d data from cpd 
#     manip_icp = SPD_from_8d(manip_icp) #list


# n_points=manip_groundtruth.shape[0]
# scaling_factor_plot = 0.3
# plot_every_nth = 1

# COLS=['s','x','y','z']

# cnt=0
# mse_naive=0.0
# mse_icp=0.0

# le_mse_naive=0.0
# le_mse_icp=0.0

# print("Errors between groundtruth and naive/icp")
# print(manip_groundtruth.shape)
# print(manip_naive.shape)
# print(manip_icp.shape)
# print(manip_input.shape)
# for i in np.arange(0,n_points,plot_every_nth):
#     m_i = manip_groundtruth[i,1:].reshape(3,3)
#     m_i_n = manip_naive[i,:].reshape(3,3)
#     m_i_icp = manip_icp[i,:].reshape(3,3)
#     m_i_in = manip_input[i,1:].reshape(3,3)

#     print("%.3f / %.3f" %(distance_riemann(m_i, m_i_n), distance_riemann(m_i, m_i_icp)))
#     mse_icp += distance_riemann(m_i, m_i_icp)**2
#     mse_naive+= distance_riemann(m_i, m_i_n)**2

#     le_mse_icp += get_logeuclidean_distance(m_i, m_i_icp)**2
#     le_mse_naive+= get_logeuclidean_distance(m_i, m_i_n)**2

#     m_i=scaling_factor_plot*m_i
#     m_i_n=scaling_factor_plot*m_i_n
#     m_i_in=scaling_factor_plot*m_i_in
#     m_i_icp=scaling_factor_plot*m_i_icp

#     X2,Y2,Z2 = get_cov_ellipsoid(m_i, [1*cnt,0,0], 1)
#     ax.plot_wireframe(X2,Y2,Z2, color='blue', alpha=0.05)

#     X2,Y2,Z2 = get_cov_ellipsoid(m_i_n, [1*cnt,0,0], 1)
#     ax.plot_wireframe(X2,Y2,Z2, color='red', alpha=0.05)

#     X2,Y2,Z2 = get_cov_ellipsoid(m_i_icp, [1*cnt,0,0], 1)
#     ax.plot_wireframe(X2,Y2,Z2, color='orange', alpha=0.05)

#     X2,Y2,Z2 = get_cov_ellipsoid(m_i_in, [1*cnt,0,0], 1)
#     ax.plot_wireframe(X2,Y2,Z2, color='green', alpha=0.05)

#     cnt+=1

# print("MSE (riemann): %.3f / %.3f" %(mse_naive/n_points, mse_icp/n_points))
# print("MSE (LogEuc): %.3f / %.3f" %(le_mse_naive/n_points, le_mse_icp/n_points))
# plt.title("MSE (riemann): %.3f / %.3f" %(mse_naive/n_points, mse_icp/n_points))

# # ax.set_zlim(-1, 1)
# # plt.xlim(-1 ,1)
# # plt.ylim(-1, 1)
# # plt.show()



# scale=np.diag([cnt, 1, 1, 1.0])
# scale=scale*(1.0/scale.max())
# scale[3,3]=0.7
# def short_proj():
#   return np.dot(Axes3D.get_proj(ax), scale)


# blue_patch = mpatches.Patch(color='blue', label='Ground Truth')
# red_patch = mpatches.Patch(color='red', label='Naive')
# orange_patch = mpatches.Patch(color='orange', label='ICP')
# green_patch = mpatches.Patch(color='green', label='Input')
# plt.legend(handles=[ blue_patch, red_patch, orange_patch, green_patch])

# ax.get_proj=short_proj
# ax.set_box_aspect(aspect = (1,1,1))

# plt.xlim(-0.5, n_points/plot_every_nth)
# plt.ylim(-0.5, 0.5)
# ax.set_zlim(-0.5, 0.5)
# #plt.ylim(-0.5, n_points/plot_every_nth)
# #ax.set_zlim(-0.5, n_points/plot_every_nth)

# results_path = "/".join(str.split(filename_manip_mapped_icp, "/")[:-1])
# robot_teacher= str.split(filename_manips_input, "/")[-3]
# robot_student= str.split(filename_manip_mapped_icp, "/")[-3]
# plt.savefig(results_path+"/mapping_from_"+robot_teacher+"_to_"+robot_student+"_plot.pdf")
# plt.show()

