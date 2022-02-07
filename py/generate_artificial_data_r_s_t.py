import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pandas as pd
from numpy import genfromtxt
from get_cov_ellipsoid import get_cov_ellipsoid, scale_volume, get_volume, scale_axes
from mpl_toolkits.mplot3d import Axes3D
import os
import matplotlib.patches as mpatches
import argparse
from pyriemann.utils.base import invsqrtm, sqrtm, logm, expm, powm
from scipy.linalg import eigh
import random
from rpa.helpers.transfer_learning.utils import mean_riemann, parallel_transport_covariances


def gen_symm(n):
    A = np.random.randn(n,n)
    return A + A.T

def gen_spd(n):
    A = gen_symm(n)
    w,v = eigh(A)
    w = np.diag(np.random.rand(len(w)))
    return np.dot(v, np.dot(w, v.T))

def gen_orth(n):
    A = gen_symm(n)
    _,Q = eigh(A)
    return Q

def perform_transformation(data, T, R, s):
    # recenter to id
    target = np.stack([np.dot(invsqrtm(mean_riemann(data)), np.dot(ti, invsqrtm(mean_riemann(data)))) for ti in data])
    # stretch
    target = np.stack([powm(covi, s) for covi in target])
    ### move target to id (slightly changed due to stretching) ###
    target = np.stack([np.dot(invsqrtm(mean_riemann(target)), np.dot(ti, invsqrtm(mean_riemann(target)))) for ti in target])  # move target to id
    # rotate
    target = np.stack([np.dot(R, np.dot(t, R.T)) for t in target])
    # recenter at T
    target = np.stack([np.dot(sqrtm(T), np.dot(ti, sqrtm(T))) for ti in target])
    return target

# r,s,t on manifold


parser = argparse.ArgumentParser()
parser.add_argument("base_path", help="base_path.", type=str)
parser.add_argument("dataset", help="dataset.", type=str) # random teacher dataset used as basis for artificial data
parser.add_argument("dataset_map", help="dataset_map", type=str) # to be mapped data for generation of ground truth
parser.add_argument('-cv', nargs='?', type=str) 

args = parser.parse_args()

R = gen_orth(3)
#T = gen_spd(3)
#s = random.uniform(0.3, 1.7)

s = 0.813 
#s=1.0

T = np.array([[0.27200416, 0.06000769, 0.17850028],
 [0.06000769, 0.53767129, 0.13022352],
 [0.17850028, 0.13022352, 0.8615307 ]])
 

#Looked good R2 <--
R = np.array([[-0.35047963,  0.87345198, -0.33800246],
 [-0.7128812,  -0.48285521, -0.50857767],
 [ 0.60742442, -0.06270949, -0.79189841]])

#2 degrees around x axis
#R=np.array([[1,0,0],
# [0, -0.41614683654,-0.90929742682],
# [0,0.90929742682,-0.41614683654]])


#R=np.eye(3)

print("Using parameters:\n s = %.3f \n" %(s))
print("Using parameters T:\n ")
print(T)
print("\nUsing parameters R:\n ")
print(R)

base_path=args.base_path
robot_teacher="panda" # based on which robot should the artificial data be created
robot_student="toy_data" # name of the generated dataset

dataset_random=args.dataset 
dataset_map=args.dataset_map # dataset which should be mapped, e.g. reach_up

filename_manip = base_path+"/"+robot_teacher+"/"+dataset_random+"/manipulabilities.csv"
filename_manip_student = base_path+"/"+robot_student+"/"+dataset_random+"/manipulabilities.csv"

if not os.path.exists(base_path+"/"+robot_student+"/"+dataset_random):
    os.makedirs(base_path+"/"+robot_student+"/"+dataset_random)

manip_tmp = genfromtxt(filename_manip, delimiter=',')
manip_tmp = manip_tmp.reshape(manip_tmp.shape[0],3,3)
manip_artificial_array=np.zeros((manip_tmp.shape[0], 9))

manip_artificial = perform_transformation(manip_tmp, T, R, s)
manip_artificial_array = manip_artificial.reshape([manip_artificial.shape[0],9])

np.savetxt(filename_manip_student, manip_artificial_array, delimiter=",")



# Generate toy data from the reach up manipulabilities from panda for testing the mapping
# only when not doing cross validation
if args.cv is None:
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    plt.title('Artificial data set with scaling: %.3f ' %(s))

    filename_manip = base_path+"/"+robot_teacher+"/"+dataset_map+"/manipulabilities_interpolated.csv"
    filename_manip_groundtruth = base_path+"/"+robot_student+"/"+dataset_map+"/manipulabilities_interpolated_groundtruth.csv"

    filename_manip_norm = base_path+"/"+robot_teacher+"/"+dataset_map+"/manipulabilities_interpolated_normalized.csv"
    filename_manip_groundtruth_norm = base_path+"/"+robot_student+"/"+dataset_map+"/manipulabilities_interpolated_groundtruth_normalized.csv"

    manip_tmp = genfromtxt(filename_manip, delimiter=',')
    manip_artificial_array=np.zeros((manip_tmp.shape[0], 10))
    manip_artificial_array[:,0]= manip_tmp[:,0]

    manip_tmp_norm_array=np.zeros((manip_tmp.shape[0], 10))
    manip_tmp_norm_array[:,0]= manip_tmp[:,0]
    manip_artificial_norm_array=np.zeros((manip_tmp.shape[0], 10))
    manip_artificial_norm_array[:,0]= manip_tmp[:,0]

    manip_tmp=manip_tmp[:,1:]
    print(manip_tmp.shape)
    manip_tmp = manip_tmp.reshape((manip_tmp.shape[0], 3, 3))

    n_points=manip_tmp.shape[0]
    scaling_factor_plot = 0.5
    plot_every_nth = 1

    COLS=['s','x','y','z']

    manip=list()
    manip_artificial_ls=list()

    manip_artificial = perform_transformation(manip_tmp, T, R, s)


    for i in np.arange(manip_tmp.shape[0]):
        manip_tmp_norm_array[i,1:] = scale_volume(manip_tmp[i],1/get_volume(manip_tmp[i])).reshape(1,9)

    for i in np.arange(manip_artificial.shape[0]):
        manip_artificial_array[i,1:] = manip_artificial[i].reshape(1,9)
        manip_artificial_norm_array[i,1:] = scale_volume(manip_artificial[i],1/get_volume(manip_artificial[i])).reshape(1,9)
        manip.append(manip_tmp[i])
        manip_artificial_ls.append(manip_artificial[i])


    np.savetxt(filename_manip_groundtruth, manip_artificial_array, delimiter=",")
    np.savetxt(filename_manip_norm, manip_tmp_norm_array, delimiter=",")
    np.savetxt(filename_manip_groundtruth_norm, manip_artificial_norm_array, delimiter=",")

    cnt=0
    for i in np.arange(0,len(manip),plot_every_nth):
        m_i = manip[i]
        m_a = manip_artificial_ls[i]

        X2,Y2,Z2 = get_cov_ellipsoid(m_i, [1*cnt,0,0], 1)
        ax.plot_wireframe(X2,Y2,Z2, color='green', alpha=0.05)

        X2,Y2,Z2 = get_cov_ellipsoid(m_a, [1*cnt,0,0], 1)
        ax.plot_wireframe(X2,Y2,Z2, color='blue', alpha=0.05)
        cnt+=1


    scale=np.diag([cnt, 1, 1, 1.0])
    scale=scale*(1.0/scale.max())
    scale[3,3]=0.7
    def short_proj():
      return np.dot(Axes3D.get_proj(ax), scale)


    ax.get_proj=short_proj
    ax.set_box_aspect(aspect = (1,1,1))

    blue_patch = mpatches.Patch(color='green', label='Original data')
    red_patch = mpatches.Patch(color='blue', label='Artificial data')
    plt.legend(handles=[ blue_patch, red_patch])

    plt.xlim(-0.5, n_points/plot_every_nth)
    plt.ylim(-0.5, 0.5)
    ax.set_zlim(-0.5, 0.5)
    #plt.ylim(-0.5, n_points/plot_every_nth)
    #ax.set_zlim(-0.5, n_points/plot_every_nth)
    #plt.show()
    plt.savefig(base_path+"/"+robot_student+"/"+dataset_map+"/plot.pdf")


