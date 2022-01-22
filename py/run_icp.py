import numpy as np
import matplotlib.pyplot as plt
from pyriemann.utils.distance import distance_riemann, distance_wasserstein
#from SPDMetrics import distance_riemann, distance_wasserstein
from pyriemann.utils.base import invsqrtm, sqrtm, logm, expm, powm
from rpa.helpers.transfer_learning.utils import mean_riemann, parallel_transport_covariances
from get_rotation_matrix import get_rotation_matrix
#from rpa.helpers.transfer_learning.manopt import get_rotation_matrix
from tqdm import tqdm
from numpy import linalg as LA
from plot_2d_embeddings import embed_2d, embed_2d_target_new, embed_2d_target_new_and_naive, plot_diffusion_embedding, \
    plot_diffusion_embedding_target_new, plot_diffusion_embedding_target_new_and_naive
import tkinter
import copy
import matplotlib
import math

import argparse


matplotlib.use('TkAgg')

'''
Based on the following paper: Riemannian Procrustes Analysis
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8588384
'''

# # nns[i] is idx for which target_idx is closest to source_i 
# def find_nearest_neighbors(source, target, distance='rie'):
#     nns = np.zeros(source.shape[0], dtype=np.int)  # for each target search smallest source
#     duplicates = np.zeros(source.shape[0], dtype=np.int) 
#     cnt = 0
#     dst_sum = 0
#     for s in source:
#         dist_min = 9999
#         idx_s = 0
#         for t in target:
#             if distance == 'rie':
#                 ds = distance_riemann(t, s)
#             elif distance == 'fro':
#                 ds = LA.norm(logm(invsqrtm(t) * s * invsqrtm(t)), 'fro')
#             elif distance == 'was':
#                 ds = distance_wasserstein(t,s)
#             else:
#                 print("[ERROR] either rie, was or fro!")
#                 return nns
#             if ds < dist_min:
#                 dist_min = ds
#                 nns[cnt] = int(idx_s)
#             idx_s += 1
#         duplicates[nns[cnt]]+=1
#         dst_sum += dist_min
#         cnt += 1
#     mean_dist = float(dst_sum / target.shape[0])
#     print("Mean dist between nns found (before removing duplicates): ", mean_dist)

#     # remove duplicates
#     print("Found %i duplicates" %(np.count(duplicates>1)))
#     for t in duplicates:
#         if t > 1:
#             stmp=np.where(nns == t)
#             for s in np.arange(stmp.shape[0]):
#                 if stmp[s]==1 :
#                     stmp[s] = distance_riemann(source[s], target[t])
#             m = min(i for i in stmp if stmp > 0)
#             smallest_dist_idx = stmp.index(m)
#             mask = np.ones(nns.shape[0])
#             mask[smallest_dist_idx]=0
#             nns[np.where(nns==t) and mask] = -1


#     print("Number of points in source included: ", np.unique(nns).size)
#     return nns, mean_dist


# # rearrange target according to nns
# def make_pairs(source, target, nns):
#     nns = nns[nns >-1]
#     target_rearranged = np.zeros(nns.shape[0])
#     for i in np.arange(nns.shape[0]):
#         target_rearranged[i] = target[nns[i]]
#     return target_rearranged


# rearrange target according to nns and duplicates removal
def make_pairs(source, target, nns):
    target_rearranged = np.zeros(target.shape)
    source_rearranged = np.zeros(target.shape)
    pos=0
    for i in np.arange(target.shape[0]):
        if nns[i] != -1:
            target_rearranged[pos] = target[nns[i]]
            source_rearranged[pos] = source[i]
            pos=pos+1

    return source_rearranged[:pos], target_rearranged[:pos]


# find nns with removing duplicates
# nns[i] is idx for which target_idx is closest to source_i 
def find_nearest_neighbors(source, target, distance='rie'):
    nns = np.zeros(source.shape[0], dtype=np.int)  # for each target search smallest source
    duplicates = np.zeros(source.shape[0], dtype=np.int) 
    cnt = 0
    dst_sum = 0
    for s in source:
        dist_min = 9999
        idx_s = 0
        for t in target:
            if distance == 'rie':
                ds = distance_riemann(t, s)
            elif distance == 'fro':
                ds = LA.norm(logm(invsqrtm(t) * s * invsqrtm(t)), 'fro')
            elif distance == 'was':
                ds = distance_wasserstein(t,s)
            else:
                print("[ERROR] either rie, was or fro!")
                return nns
            if ds < dist_min:
                dist_min = ds
                nns[cnt] = int(idx_s)
            idx_s += 1
        duplicates[nns[cnt]]+=1
        dst_sum += dist_min
        cnt += 1
    mean_dist = float(dst_sum / target.shape[0])
    print("Mean dist between nns found (before removing duplicates): ", mean_dist)

    # remove duplicates
    print("Found %i duplicates and %i unique target sample neighbors" %(np.argwhere(duplicates>1).shape[0], np.unique(nns).shape[0]))

    for i in np.arange(duplicates.shape[0]):
        if duplicates[i] > 1:
            stmp=np.argwhere(nns==i).ravel()
            d=np.zeros(stmp.shape[0])
            for s_idx in np.arange(stmp.shape[0]):
                d[s_idx] = distance_riemann(source[stmp[s_idx]], target[i])

            min_idx= np.argmin(d)
            stmp = np.delete(stmp, min_idx)
            nns[stmp] = -1

    source, target = make_pairs(source, target, nns)

    mean_dist_2 = 0.0
    for (s,t) in zip(source, target):
        mean_dist_2+= distance_riemann(s,t)
    mean_dist_2 = mean_dist_2/target.shape[0]

    print("Mean dist between nns found (after removing duplicates): ", mean_dist_2)

    return source, target, mean_dist_2

# assumes make_pairs already called!
def reject_max_distance(source, target, max_dist, dist='rie'):
    idx_rm=list()
    if dist=='rie':
        for i in np.arange(source.shape[0]):
            if distance_riemann(source[i], target[i]) > max_dist:
                idx_rm.append(i)
    if dist=='was':
        for i in np.arange(source.shape[0]):
            if distance_wasserstein(source[i], target[i]) > max_dist:
                idx_rm.append(i)
    source = np.delete(source, idx_rm, 0)
    target = np.delete(target, idx_rm, 0)
    return source, target


# assumes make_pairs already called!
def reject_median(source,target, dist='rie'):
    distances=np.zeros(source.shape[0])
    if dist=='rie':
        for i in np.arange(source.shape[0]):
            distances[i] = distance_riemann(source[i], target[i])
    if dist=='was':
        for i in np.arange(source.shape[0]):
            distances[i] = distance_wasserstein(source[i], target[i])
    idx = np.argsort(distances) # indices from small to large
    idx = idx[:int(idx.shape[0]/2)] # only keep lower half
    source = source[idx]
    target = target[idx]
    return source, target


def get_mean_dist_nns(source, target, distance='rie'):
    dst_sum = 0
    for t, s in zip(target, source):
        if distance == 'rie':
            ds = distance_riemann(t, s)
        elif distance == 'fro':
            ds = LA.norm(logm(invsqrtm(t) * s * invsqrtm(t)), 'fro')
        elif distance == 'was':
            ds = distance_wasserstein(t,s)
        else:
            print("[ERROR] either riem or fro!")
            return -1
        dst_sum += ds
    mean_dist = float(dst_sum / target.shape[0])
    print("Mean dist between source and target pairs given: ", mean_dist)
    return mean_dist


# for initial step we subsample the most farthest points from mean
def find_subsample_idx(source, target):
    num_points = math.ceil(target.shape[0]*0.1)
    if(num_points<11 and target.shape[0]>=50):
        num_points=10
    if(num_points<3 and target.shape[0]<=20):
        num_points=3
    print("For initial NNS %i points are used" %(num_points))
    idx_t = (np.array([distance_riemann(covi, np.eye(3)) ** 2 for covi in target])).argsort()[-num_points:][::-1] # find num_points points with biggest distance to id
    idx_s = (np.array([distance_riemann(covi, np.eye(3)) ** 2 for covi in source])).argsort()[-num_points:][::-1] # find num_points points with biggest distance to id
    return idx_s, idx_t


# move to id, scale, then find nns then rotate
def initial_iteration(source_org, target, results_path, dist):
    mean_target = mean_riemann(target)
    mean_source = mean_riemann(source_org)

    fig2 = plt.figure(figsize=(20, 7))
    fig2.suptitle('Initial iteration process')
    axs2 = list()
    axs2.append(fig2.add_subplot(1, 5, 1))
    axs2[0].set_title("Original")
    plot_diffusion_embedding(source_org, target, axs2[0])

    ### move target and source to id ###
    target = np.stack([np.dot(invsqrtm(mean_target), np.dot(ti, invsqrtm(mean_target))) for ti in target])  # move target to id
    source = np.stack([np.dot(invsqrtm(mean_source), np.dot(si, invsqrtm(mean_source))) for si in source_org])  # move source to id

    axs2.append(fig2.add_subplot(1, 5, 2))
    axs2[1].set_title("Recenter to id")
    plot_diffusion_embedding(source, target, axs2[1])

    ### stretch target at id ###
    disp_source = np.sum([distance_riemann(covi, np.eye(3)) ** 2 for covi in source]) / len(source)  # get stretching factor
    disp_target = np.sum([distance_riemann(covi, np.eye(3)) ** 2 for covi in target]) / len(target)
    s = 1.0 / round(np.sqrt(disp_target / disp_source),4)
    print("Disp source=%.2f, disp target= %.2f, s= %.2f" %(disp_source, disp_target,s))
    target = np.stack([powm(ti, s) for ti in target])  # stretch target at id

    ### move target to id (slightly changed due to stretching) ###
    #target = np.stack([np.dot(invsqrtm(mean_riemann(target)), np.dot(ti, invsqrtm(mean_riemann(target)))) for ti in target])  # move target to id

    # subsampling 
    idx_s, idx_t = find_subsample_idx(source, target)

    axs2.append(fig2.add_subplot(1, 5, 3))
    axs2[2].set_title("Stretch at id")
    plot_diffusion_embedding(source, target, axs2[2], idx_s, idx_t)

    target_subsample_id = target[idx_t]
    source_subsample_id = source[idx_s]

    # find nns in subsamples
    source_subsample_id, target_subsample_id, _ = find_nearest_neighbors(source_subsample_id, target_subsample_id, dist)
    #target_subsample_id = make_pairs(source_subsample_id, target_subsample_id, nns) # rearrange target acc to nns

    ### find rotation with subsamples only ###
    R = get_rotation_matrix(M=source_subsample_id, Mtilde=target_subsample_id, weights=None, dist=dist)
    #R = get_rotation_matrix(M=source_subsample_id, Mtilde=target_subsample_id, weights=None, dist='rie', x=np.eye(3))
    #print(R[1])
    R=R[0] # tuple due to verboselevel=2 in solve()
    print("Rotation found: ", R)
    target = np.stack([np.dot(R, np.dot(t, R.T)) for t in target]) # apply rotation to all samples
    #target = np.stack([R.T @ t @ R for t in target]) # apply rotation to all samples

    axs2.append(fig2.add_subplot(1, 5, 4))
    axs2[3].set_title("Rotate wrt subsamples")
    plot_diffusion_embedding(source, target, axs2[3], idx_s, idx_t)

    ### move to source ###
    target = np.stack([np.dot(sqrtm(mean_source), np.dot(ti, sqrtm(mean_source))) for ti in target]) 

    axs2.append(fig2.add_subplot(1, 5, 5))
    axs2[4].set_title("Move to source")
    plot_diffusion_embedding(source_org, target, axs2[4], idx_s, idx_t)
    fig2.savefig(results_path+"/icp_initial_iteration.pdf", bbox_inches='tight')

    return np.ravel(R), np.ravel(s), np.ravel(mean_source), target



def recenter_to_id_stretch_rotate_recenter_to_source(source_org, target, calculate_stretching_factor=True, s=1):
    mean_target = mean_riemann(target)
    mean_source_org = mean_riemann(source_org)

    ### move target to id ###
    target = np.stack([np.dot(invsqrtm(mean_target), np.dot(ti, invsqrtm(mean_target))) for ti in target])  
    source = np.stack([np.dot(invsqrtm(mean_source_org), np.dot(si, invsqrtm(mean_source_org))) for si in source_org])  # move source to id

    ### stretch target at id ###
    if calculate_stretching_factor:    
        disp_source = np.sum([distance_riemann(covi, np.eye(3)) ** 2 for covi in source]) / len(source)  # get stretching factor
        disp_target = np.sum([distance_riemann(covi, np.eye(3)) ** 2 for covi in target]) / len(target)
        s = 1.0 / round(np.sqrt(disp_target / disp_source),4)
        target = np.stack([powm(ti, s) for ti in target])  # stretch target at id

    ### move target to id (slightly changed due to stretching) ###
    target = np.stack([np.dot(invsqrtm(mean_riemann(target)), np.dot(ti, invsqrtm(mean_riemann(target)))) for ti in target])  # move target to id

    ### rotate ###
    target, rotation_matrix_iter = rotate(source, target)

    ### move to source ###
    target = np.stack([np.dot(sqrtm(mean_source_org), np.dot(ti, sqrtm(mean_source_org))) for ti in target]) 

    return target, s, rotation_matrix_iter


def rotate(source, target, weights=None, distance='rie'):  # rie or euc
    weights=np.ones(target.shape[0])
    #R = get_rotation_matrix(M=source, Mtilde=target, weights=weights, dist=distance)
    R = get_rotation_matrix(M=source, Mtilde=target, weights=weights, dist=distance, x=np.eye(3))
    print(R[1])
    R=R[0] # tuple due to verboselevel=2 in solve()
    print(R)
    target = np.stack([np.dot(R, np.dot(t, R.T)) for t in target])
    #target_rotated = np.stack([R.T @ t @ R for t in target])
    return target, R


def perform_transformation(target, T, R, s):
    assert(len(T)==len(R)==len(s))

    # recenter to id
    mean_target = mean_riemann(target)
    target = np.stack([np.dot(invsqrtm(mean_target), np.dot(ti, invsqrtm(mean_target))) for ti in target])

    for i in np.arange(len(R)):
        # stretch
        target = np.stack([powm(covi, s[i]) for covi in target])
        ### move target to id (slightly changed due to stretching) ###
        target = np.stack([np.dot(invsqrtm(mean_riemann(target)), np.dot(ti, invsqrtm(mean_riemann(target)))) for ti in target])  # move target to id
        # rotate
        target = np.stack([np.dot(R[i], np.dot(t, R[i].T)) for t in target])
        #target = np.stack([R[i].T @ t @ R[i] for t in target])

    # recenter to source (all Ti same)
    target = np.stack([np.dot(sqrtm(T[-1]), np.dot(ti, sqrtm(T[-1]))) for ti in target])
    return target



parser = argparse.ArgumentParser()
parser.add_argument("base_path", help="base_path.", type=str)
parser.add_argument("robot_teacher", help="robot_teacher.", type=str)
parser.add_argument("robot_student", help="robot_student", type=str)
parser.add_argument("lookup_dataset", help="path to lookup dataset e.g. 5000", type=str)
parser.add_argument("map_run", help="0: only find mapping params, 1: only new points, 2: both, 3: only new points no plot", type=int) # cv also 2
parser.add_argument("--map_dataset", help="path to data to be mapped", type=str, nargs='?') # only when not cv
parser.add_argument("--cv_k", help="cv_k", type=int, nargs='?')

args = parser.parse_args()

if(args.map_run==0):
    run_map_find = True
    run_map_new = False 
if(args.map_run==1):
    run_map_find = False
    run_map_new = True 
if(args.map_run==2):
    run_map_find = True
    run_map_new = True 
if(args.map_run==3):
    run_map_find = False
    run_map_new = False

iter = 1
dist = "rie"

source = np.genfromtxt(args.base_path+"/"+args.robot_student+"/"+args.lookup_dataset+"/manipulabilities.csv", delimiter=',')
target = np.genfromtxt(args.base_path+"/"+args.robot_teacher+"/"+args.lookup_dataset+"/manipulabilities.csv", delimiter=',')

source_org = source.reshape((source.shape[0], 3, 3))
target = target.reshape((target.shape[0], 3, 3))

results_path = args.base_path+"/"+args.robot_teacher+"/"+args.lookup_dataset
results_path2 = args.base_path+"/"+args.robot_student+"/"+args.lookup_dataset

#if cross validation
if args.cv_k is not None:
    idx = np.genfromtxt(args.base_path+"/"+args.robot_student+"/"+args.lookup_dataset+"/cv/cv_idx.csv", delimiter=',')
    manip_groundtruth=source_org[np.where(idx[args.cv_k,:]==1)]
    source_org=source_org[np.where(idx[args.cv_k,:]==0)]
    target_new=target[np.where(idx[args.cv_k,:]==1)]
    target=target[np.where(idx[args.cv_k,:]==0)]
    results_path = args.base_path+"/"+args.robot_teacher+"/"+args.lookup_dataset+"/cv"
    results_path2 = args.base_path+"/"+args.robot_student+"/"+args.lookup_dataset+"/cv"


### the alignment is performed s.t. target matches source afterwards, so target is rotated, source not
if run_map_find:

    R = np.zeros((iter, 3, 3))
    T = np.zeros((iter, 3, 3))
    s = np.zeros(iter)


    fig2 = plt.figure(figsize=(20, 7))
    fig2.suptitle('Diffusion Map')
    axs2 = list()
    axs2.append(fig2.add_subplot(1, iter + 2, 1))
    axs2[0].set_title("Original")
    plot_diffusion_embedding(source_org, target, axs2[0])

    if iter==0:
        R,s,T,target = initial_iteration(source_org, target, results_path2, dist)
    else:
        _,_,_,target = initial_iteration(source_org, target, results_path2, dist)

    ### find nns ###
    source, target, _ = find_nearest_neighbors(source_org, target, dist)

    axs2.append(fig2.add_subplot(1, iter + 2, 2))
    axs2[1].set_title("Initial (%.2f)" %(get_mean_dist_nns(source, target)))
    plot_diffusion_embedding(source_org, target, axs2[1])

    #target = make_pairs(source_org, target, nns) # rearrange target acc to nns s.t. s[i] is closest point to t[i] --> not all ti included!

    print("-------------------------------------------------------------")

    for iter_i in tqdm(np.arange(iter)):
        print("Iteration %i/%i" %(iter_i, iter))

        target, s_iter, rotation_matrix_iter = recenter_to_id_stretch_rotate_recenter_to_source(source, target)

        source, target, mean_dist_nns = find_nearest_neighbors(source_org, target, dist)
        #target = make_pairs(source_org, target, nns)
    
        R[iter_i] = rotation_matrix_iter
        s[iter_i] = s_iter
        T[iter_i] = mean_riemann(source_org)

        print("R: ", R[iter_i])
        print("s: ", s[iter_i])
        print("T: ", T[iter_i])

        axs2.append(fig2.add_subplot(1, iter + 2, iter_i + 3))
        plot_diffusion_embedding(source_org, target, axs2[iter_i + 2])
        axs2[iter_i + 2].set_title("%.2f" % (mean_dist_nns))

        print("-------------------------------------------------------------")

    # save results
    if iter==0:
        np.savetxt(results_path+"/R_icp_"+args.robot_teacher+"_to_"+args.robot_student+".txt", np.reshape(R, (1, 9)), delimiter=',')
        np.savetxt(results_path+"/T_icp_"+args.robot_teacher+"_to_"+args.robot_student+".txt", np.reshape(T, (1, 9)), delimiter=',')
        np.savetxt(results_path+"/s_icp_"+args.robot_teacher+"_to_"+args.robot_student+".txt", s, delimiter=',')
    else:
        np.savetxt(results_path+"/R_icp_"+args.robot_teacher+"_to_"+args.robot_student+".txt", np.reshape(R, (iter, 9)), delimiter=',')
        np.savetxt(results_path+"/T_icp_"+args.robot_teacher+"_to_"+args.robot_student+".txt", np.reshape(T, (iter, 9)), delimiter=',')
        np.savetxt(results_path+"/s_icp_"+args.robot_teacher+"_to_"+args.robot_student+".txt", s, delimiter=',')

    fig2.tight_layout()
    fig2.savefig(results_path2+"/icp_diffusion_map.pdf", bbox_inches='tight')



# apply trafo to new points
if run_map_new:
    if args.cv_k is None:
        target_new = np.genfromtxt(args.base_path+"/"+args.robot_teacher+"/"+args.map_dataset+"/manipulabilities_interpolated.csv", delimiter=',')
        target_new = target_new[1:,1:].reshape((target_new[1:,1:].shape[0], 3, 3))   
        filename_manip_groundtruth=args.base_path+"/"+args.robot_student+"/"+args.map_dataset+"/manipulabilities_interpolated_groundtruth.csv" 
        manip_groundtruth = np.genfromtxt(filename_manip_groundtruth, delimiter=',')
        manip_groundtruth = manip_groundtruth[:,1:].reshape((manip_groundtruth[:,1:].shape[0], 3, 3))   

    R = np.genfromtxt(results_path+"/R_icp_"+args.robot_teacher+"_to_"+args.robot_student+".txt", delimiter=',')
    T = np.genfromtxt(results_path+"/T_icp_"+args.robot_teacher+"_to_"+args.robot_student+".txt", delimiter=',')
    s = np.genfromtxt(results_path+"/s_icp_"+args.robot_teacher+"_to_"+args.robot_student+".txt", delimiter=',')
    if iter==0:
        R = np.reshape(R, (1, 3, 3))
        T = np.reshape(T, (1, 3, 3))
    else:
        R = np.reshape(R, (iter, 3, 3))
        T = np.reshape(T, (iter, 3, 3))
    s=s.ravel()
    # print(s)
    # print(T)
    # print(R)


    fig2 = plt.figure(figsize=(16.5, 6))
    axs2 = list()
    axs2.append(fig2.add_subplot(1, 2, 1))
    axs2[0].set_title("Input before transformation")
    plot_diffusion_embedding_target_new(source_org, target, target_new, axs2[0])

    target_new = perform_transformation(target_new, T, R, s)

    if args.cv_k is None:
        np.savetxt(args.base_path+"/"+args.robot_student+"/"+args.map_dataset+"/manipulabilities_interpolated_mapped_icp.csv", np.reshape(target_new, (target_new.shape[0], 9)), delimiter=",")
        target_mapped_naive = np.genfromtxt(args.base_path+"/"+args.robot_student+"/"+args.map_dataset+"/manipulabilities_interpolated_mapped_naive.csv", delimiter=',', dtype=np.double)
    else:
        np.savetxt(args.base_path+"/"+args.robot_student+"/"+args.lookup_dataset+"/cv/manipulabilities_mapped_icp.csv", np.reshape(target_new, (target_new.shape[0], 9)), delimiter=",")
        target_mapped_naive = np.genfromtxt(args.base_path+"/"+args.robot_student+"/"+args.lookup_dataset+"/cv/manipulabilities_mapped_naive.csv", delimiter=',', dtype=np.double)
    
    axs2.append(fig2.add_subplot(1, 2, 2))

    #plot naive mapping too in 2d embedding to be able to compare results
    target_mapped_naive = target_mapped_naive.reshape((target_mapped_naive.shape[0], 3, 3))
    for i in target:
        assert(np.all(np.linalg.eigvals(i) > 0))
    for i in target_new:
        assert(np.all(np.linalg.eigvals(i) > 0))
    for i in target_mapped_naive:
        assert(np.all(np.linalg.eigvals(i) > 0))

    axs2[1].set_title("Input after transformation (Naive/ICP)")
    plot_diffusion_embedding_target_new_and_naive(manip_groundtruth, target, target_new, target_mapped_naive, axs2[1])
    #plot_diffusion_embedding_target_new_and_naive(source_org, target, target_new, target_mapped_naive, axs2[1])

    fig2.tight_layout()
    fig2.savefig(results_path2+"/mapped_diffusion_map.pdf")












# if (run_map_new==False and run_map_find==False):
#     results_path = args.base_path+"/"+args.robot_student+"/"+args.map_dataset
#     target_new = np.genfromtxt(args.base_path+"/"+args.robot_teacher+"/"+args.map_dataset+"/manipulabilities_interpolated.csv", delimiter=',')
#     R = np.genfromtxt(results_path+"/R_icp_"+args.robot_teacher+"_to_"+args.robot_student+".txt", delimiter=',')
#     T = np.genfromtxt(results_path+"/T_icp_"+args.robot_teacher+"_to_"+args.robot_student+".txt", delimiter=',')
#     s = np.genfromtxt(results_path+"/s_icp_"+args.robot_teacher+"_to_"+args.robot_student+".txt", delimiter=',')
#     R = np.reshape(R, (iter, 3, 3))
#     T = np.reshape(T, (iter, 3, 3))

#     #source = source.reshape((source.shape[0], 3, 3))
#     #target = target.reshape((target.shape[0], 3, 3))
#     target_new = target_new.reshape((target_new.shape[0], 3, 3))


#     #print(R)
#     #print(T)
#     #print(s)

#     # fig1 = plt.figure(figsize=(16.5, 6))
#     # fig2 = plt.figure(figsize=(16.5, 6))
#     # axs1 = list()
#     # axs2 = list()
#     # axs1.append(fig1.add_subplot(1, 2, 1))
#     # axs2.append(fig2.add_subplot(1, 2, 1))
#     # axs1[0].set_title("TSNE")
#     # axs2[0].set_title("Diffusion")
#     # axs2[0].set_xlim([-5, 5])
#     # axs2[0].set_ylim([-5, 5])
#     #embed_2d_target_new(source, target, target_new, axs1[0])
#     #plot_diffusion_embedding_target_new(source, target, target_new, axs2[0])

#     target_new_rct = perform_translation_and_stretching(target_new, T[-1], s)
#     target_rct = perform_translation_and_stretching(target, T[-1], s)
#     source_rct = source

#     target_rt = perform_rotation(target_rct, R)
#     source_rt = source_rct
#     target_new_rt = perform_rotation(target_new_rct, R)

#     np.savetxt(args.base_path+"/"+args.robot_student+"/"+args.map_dataset+"/manipulabilities_interpolated_mapped_icp.csv", np.reshape(target_new_rt, (target_new_rt.shape[0], 9)))

#     #axs1.append(fig1.add_subplot(1, 2, 2))
#     #axs2.append(fig2.add_subplot(1, 2, 2))

#     for i in target_rt:
#         assert(np.all(np.linalg.eigvals(i) > 0))
#     for i in target_new_rt:
#         assert(np.all(np.linalg.eigvals(i) > 0))
#     for i in target_mapped_naive:
#         assert(np.all(np.linalg.eigvals(i) > 0))

#     #plot naive mapping too in 2d embedding to be able to compare results
#     target_mapped_naive = np.genfromtxt(args.base_path+"/"+args.robot_student+"/"+args.map_dataset+"/manipulabilities_interpolated_mapped_naive.csv", delimiter=',', dtype=np.double)
#     target_mapped_naive = target_mapped_naive.reshape((target_mapped_naive.shape[0], 3, 3)) # + 1e-3
#     print("Naive mapped points: ", target_mapped_naive.shape)

#     #embed_2d_target_new_and_naive(source_rt, target_rt, target_new_rt, target_mapped_naive, axs1[1])
#     #plot_diffusion_embedding_target_new_and_naive(source_rt, target_rt, target_new_rt, target_mapped_naive, axs2[1])

#     #embed_2d_target_new(source_rt, target_rt, target_new_rt, axs1[1])
#     #plot_diffusion_embedding_target_new(source_rt, target_rt, target_new_rt, axs2[1])

#     #plt.show()
#     #fig1.tight_layout()
#     #fig2.tight_layout()
#     #fig1.savefig(results_path+"/mapped_tsne.pdf")
#     #fig2.savefig(results_path+"/mapped_diffusion_map.pdf")
