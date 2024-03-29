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
import math
from rpa.helpers.transfer_learning.utils import mean_riemann
from pyriemann.utils.base import invsqrtm, sqrtm, logm, expm, powm
from pyriemann.utils.distance import distance_riemann

plt.rcParams['text.usetex'] = True


def get_errors(data1p, data2p):

  data1 = genfromtxt(data1p, delimiter=',')
  data2 = genfromtxt(data2p, delimiter=',')

  data2cp=data2.reshape((data2.shape[0],3,3))
  mean_data=mean_riemann(data2cp)
  data2cp = np.stack([np.dot(invsqrtm(mean_data), np.dot(ti, invsqrtm(mean_data))) for ti in data2cp])  
  disp = np.sum([distance_riemann(covi, np.eye(3)) ** 2 for covi in data2cp]) / len(data2cp)

  if "8d" in data2p: # 8d data from cpd
     data2 = SPD_from_8d(data2) #list

  if data1.shape[1]==10:
    data1=data1[:,1:]

  mse_icp=0.0
  le_mse_icp=0.0

  f = open(final_results_path+"/info.txt","w+")

  for (mi,mj) in zip(data1, data2):
    mi=mi.reshape(3,3)
    mj=mj.reshape(3,3)

    w,v = np.linalg.eigh(mi) # just for very singular cases as in trajectories generated
    w[w<1e-12]=0.01
    m=np.matmul(np.matmul(v, np.diag(w)), v.transpose())
    mi=m

    w,v = np.linalg.eigh(mj) # just for very singular cases as in trajectories generated
    w[w<1e-12]=0.01
    m=np.matmul(np.matmul(v, np.diag(w)), v.transpose())
    mj=m

    print("%.3f " %(distance_riemann(mi, mj)))
    f.write("\n%.3f " %(distance_riemann(mi, mj)))
    mse_icp += distance_riemann(mi, mj)**2
    le_mse_icp += get_logeuclidean_distance(mi, mj)**2

  n_points = data1.shape[0]
  print("nPoints=%i"%(n_points))
  print("MSE/ RMSE (riemann): %.3f/%.3f " %(mse_icp/n_points, (math.sqrt(mse_icp/n_points))))
  print("MSE/ RMSE (riemann)/disp: %.3f/%.3f " %(mse_icp/n_points, (math.sqrt(mse_icp/n_points))/disp))
  print("MSE (LogEuc): %.3f/%.3f " %(le_mse_icp/n_points, math.sqrt(le_mse_icp/n_points)))
  f.write("\n------------")
  f.write("\nMSE/ RMSE (riemann): %.3f/%.3f " %(mse_icp/n_points, (math.sqrt(mse_icp/n_points))))
  f.write("\nMSE/ RMSE (riemann)/disp: %.3f/%.3f " %(mse_icp/n_points, (math.sqrt(mse_icp/n_points))/disp))
  f.write("\nMSE (LogEuc): %.3f/%.3f " %(le_mse_icp/n_points, math.sqrt(le_mse_icp/n_points)))
  f.close()



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-mapping_paths','--l', nargs='+', type=str)

  args = parser.parse_args()
  args.paths = [item for item in args.l[0].split(',')]
  print(args.paths)

  if(len(args.paths)==2):
    labellist=['\\textit{input}','\\textit{ICP}']
    # colors=['black','mediumorchid']
    colors=['darkgray','firebrick']
  else:
    labellist=['\\textit{input}','\\textit{ground truth}','\\textit{ICP}']
    # colors=['black','darkblue', 'mediumorchid', 'plum', 'cornflowerblue']
    #colors=['darkorange', 'limegreen','rebeccapurple', 'plum', 'darkgrey']
    #colors=['darkgrey','darkblue', 'mediumorchid', 'plum', 'darkgrey']
    colors=['darkgrey','royalblue', 'firebrick', 'plum', 'darkgrey']

  scaling_factor=0.1
  plot_every_nth=3

  map_dataset=(args.paths[0]).split("/")[-2]
  print(map_dataset)
  # final_results_path="/home/nnrthmr/CLionProjects/ma_thesis/data/final_results/realworld/validation/"+map_dataset
  # final_results_path="/home/nnrthmr/CLionProjects/ma_thesis/data/final_results/toydata/validation/"+map_dataset
  # final_results_path="/home/nnrthmr/CLionProjects/ma_thesis/data/final_results/2dof-2dofvertical/specialTP/100points/validation/"+map_dataset
  final_results_path="/home/nnrthmr/CLionProjects/ma_thesis/data/final_results/newestExpHR/PT+ICP/25points"



  ### front view ###
  fig = plt.figure(figsize=(40, 15))
  fig.subplots_adjust(bottom=-0.15,top=1.2,wspace=0, hspace=0, right=1)
  #ax = plt.axes(projection='3d')
  ax = fig.gca(projection='3d')

  ### top view ###
  fig2 = plt.figure(figsize=(40, 15))
  fig2.subplots_adjust(bottom=-0.15,top=1.2,wspace=0, hspace=0, right=1)
  #ax2 = plt.axes(projection='3d')
  ax2 = fig2.gca(projection='3d')

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

    print("DATA SHAPE: ", data.shape)

    alpha=0.1

    manip=list()

    for i in np.arange(data.shape[0]):
      mm=data[i].reshape(3,3)
      manip.append(scaling_factor*mm)

    cnt=0
    for i in np.arange(0,len(manip),plot_every_nth):
      m_i = manip[i]
      w,v = np.linalg.eigh(m_i) # just for very singular cases as in trajectories generated
      w[w<1e-12]=0.001
      m=np.matmul(np.matmul(v, np.diag(w)), v.transpose())
      m_i=m
      X2,Y2,Z2 = get_cov_ellipsoid(m_i, [0,0.5*cnt,0], 1)

      # if c>0:
      #   alpha=0.2
      # if c==0:
      #   alpha=0.1
      # if c==0:
      #   alpha=0.09
      if c==0:
         alpha=0.3
      # if c==2:
      #   alpha=0.3
      # ax.plot_surface(X2,Y2,Z2, color=colors[c], alpha=alpha)
      ax.plot_wireframe(X2,Y2,Z2, color=colors[c], alpha=alpha)
      # ax2.plot_surface(X2,Y2,Z2, color=colors[c], alpha=alpha)

      # if c==0:
      #   alpha=0.035
      # if c==1:
      #   alpha=0.1
      # if c==2:
      #   alpha=0.1
      ax2.plot_wireframe(X2,Y2,Z2, color=colors[c], alpha=alpha)
      cnt+=1
    c+=1


  # print errors
  if(len(args.paths)==2):
    print("errs between 0 and 1")
    get_errors(args.paths[0],args.paths[1])
  else:
    get_errors(args.paths[1],args.paths[2])

  scale=np.diag([1, 0.5*cnt, 1, 1.0])
  scale=scale*(1.0/scale.max())
  scale[3,3]=0.7

  def short_proj():
    return np.dot(Axes3D.get_proj(ax), scale)
  def short_proj2():
    return np.dot(Axes3D.get_proj(ax2), scale)

  ax.get_proj=short_proj
  ax.set_box_aspect(aspect = (1,1,1))
  ax.view_init(azim=0, elev=0)


  blue_patch = mpatches.Patch(color='royalblue', label=labellist[1])
  # blue_patch = mpatches.Patch(color='royalblue', label=labellist[1])
  red_patch = mpatches.Patch(color='darkgrey', label=labellist[0])
  # red_patch = mpatches.Patch(color='limegreen', label=labellist[0])
  if len(args.paths)==3:
    orange_patch = mpatches.Patch(color='firebrick', label=labellist[2])
    ax.legend(handles=[ blue_patch, red_patch, orange_patch], loc='center left', bbox_to_anchor=(1.07, 0.51),fontsize=20)
    ax2.legend(handles=[ blue_patch, red_patch, orange_patch], loc='center left', bbox_to_anchor=(1.07, 0.51),fontsize=20)
  elif len(args.paths)==4:
    orange_patch = mpatches.Patch(color='plum', label=labellist[3])
    ax.legend(handles=[ blue_patch, red_patch, orange_patch], loc='center left', bbox_to_anchor=(1.07, 0.51))
    ax2.legend(handles=[ blue_patch, red_patch, orange_patch], loc='center left', bbox_to_anchor=(1.07, 0.51))
  else:
    blue_patch = mpatches.Patch(color='firebrick', label=labellist[1])
    red_patch = mpatches.Patch(color='dimgray', label=labellist[0])
    ax.legend(handles=[ blue_patch, red_patch], loc='center left', bbox_to_anchor=(1.07, 0.51))
    ax2.legend(handles=[ blue_patch, red_patch], loc='center left', bbox_to_anchor=(1.07, 0.51))

  plt.ylim(-0.5, 0.5*len(manip)/plot_every_nth+0.5)
  plt.xlim(-0.5, 0.5)
  ax.set_zlim(-0.5, 0.5)
  #ax.set_xlabel('$x$')
  ax.set_xticks([])
  ax.set_yticks([])
  ax.set_zticks([])
  ax.set_ylabel('\\textit{y (Front view)}',labelpad=40)
  ax.set_zlabel('\\textit{z}')


  ax2.get_proj=short_proj2
  ax2.set_box_aspect(aspect = (1,1,1))
  ax2.view_init(azim=0, elev=90)


  #plt.ylim(-0.5, 0.5*len(manip)/plot_every_nth+0.5)
  #plt.xlim(-0.5, 0.5)
  ax2.set_zlim(-0.5, 0.5)
  ax2.set_zticks([])
  ax2.set_xticks([])
  ax2.set_yticks([])
  ax2.set_xlabel('$x$')
  #ax2.yaxis.tick_left()
  #ax2.xaxis.set_ticks_position('both')
  ax2.xaxis.set_label_position('bottom')
  ax2.set_ylabel('\\textit{y (Top view)}', labelpad=40)

  #ax.set_title("Front view")


  fig.tight_layout()
  fig2.tight_layout()
  fig.savefig(final_results_path+"/mapped_front_view.svg")
  fig2.savefig(final_results_path+"/mapped_top_view.svg")
  fig.savefig(final_results_path+"/mapped_front_view.svg")
  fig2.savefig(final_results_path+"/mapped_top_view.svg")

  #tmp_planes = ax2.xaxis._PLANES 
  #ax2.xaxis._PLANES = ( tmp_planes[2], tmp_planes[3], 
  #                   tmp_planes[0], tmp_planes[1], 
  #                   tmp_planes[4], tmp_planes[5])
  #ax2.xaxis.set_rotate_label(False)  # disable automatic rotation

  
  #plt.show()


