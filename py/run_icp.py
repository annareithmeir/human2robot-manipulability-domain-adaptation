import numpy as np
import matplotlib.pyplot as plt
from pyriemann.utils.distance import distance_riemann
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

# nns[i] is idx for which target_idx is closest to source_i 
def find_nearest_neighbors(source, target, distance='riem'):
    nns = np.zeros(source.shape[0], dtype=np.int)  # for each target search smallest source
    cnt = 0
    dst_sum = 0
    for s in source:
        dist_min = 9999
        idx_s = 0
        for t in target:
            if distance == 'riem':
                ds = distance_riemann(t, s)
            elif distance == 'fro':
                ds = LA.norm(logm(invsqrtm(t) * s * invsqrtm(t)), 'fro')
            else:
                print("[ERROR] either riem or fro!")
                return nns
            if ds < dist_min:
                dist_min = ds
                nns[cnt] = int(idx_s)
            idx_s += 1
        cnt += 1
        dst_sum += dist_min
    mean_dist = float(dst_sum / target.shape[0])
    print("Mean dist between nns found: ", mean_dist)
    print("Number of points in source included: ", np.unique(nns).size)
    return nns, mean_dist


# rearrange target according to nns
def make_pairs(source, target, nns):
    target_rearranged = np.zeros(target.shape)
    for i in np.arange(target.shape[0]):
        target_rearranged[i] = target[nns[i]]
    return target_rearranged


def get_mean_dist_nns(source, target, distance='riem'):
    dst_sum = 0
    for t, s in zip(target, source):
        if distance == 'riem':
            ds = distance_riemann(t, s)
        elif distance == 'fro':
            ds = LA.norm(logm(invsqrtm(t) * s * invsqrtm(t)), 'fro')
        else:
            print("[ERROR] either riem or fro!")
            return -1
        dst_sum += ds
    mean_dist = float(dst_sum / target.shape[0])
    print("Mean dist between source and target pairs given: ", mean_dist)
    return mean_dist


def find_subsample_idx(source, target):
    ### subsample of most distant points ###
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
def initial_iteration(source, target, results_path, calculate_stretching_factor=True, s=1):
    mean_target = mean_riemann(target)
    mean_source = mean_riemann(source)

    fig2 = plt.figure(figsize=(20, 7))
    fig2.suptitle('Initial iteration process')
    axs2 = list()
    axs2.append(fig2.add_subplot(1, 5, 1))
    axs2[0].set_title("Original")
    plot_diffusion_embedding(source, target, axs2[0])

    ### move target and source to id ###
    target_rct_id = np.stack([np.dot(invsqrtm(mean_target), np.dot(ti, invsqrtm(mean_target))) for ti in target])  # move target to id
    source_rct_id = np.stack([np.dot(invsqrtm(mean_source), np.dot(si, invsqrtm(mean_source))) for si in source])  # move source to id

    axs2.append(fig2.add_subplot(1, 5, 2))
    axs2[1].set_title("Recenter to id")
    plot_diffusion_embedding(source_rct_id, target_rct_id, axs2[1])

    ### stretch target at id ###
    if calculate_stretching_factor:    
        disp_source = np.sum([distance_riemann(covi, np.eye(3)) ** 2 for covi in source_rct_id]) / len(source_rct_id)  # get stretching factor
        disp_target = np.sum([distance_riemann(covi, np.eye(3)) ** 2 for covi in target_rct_id]) / len(target_rct_id)
        s = 1.0 / round(np.sqrt(disp_target / disp_source),4)
        print("Disp source=%.2f, disp target= %.2f, s= %.2f" %(disp_source, disp_target,s))
        target_rct_id = np.stack([powm(ti, s) for ti in target_rct_id])  # stretch target at id


    ### move target and source to id (slightly changed due to stretching) ###
    target_rct_id = np.stack([np.dot(invsqrtm(mean_riemann(target_rct_id)), np.dot(ti, invsqrtm(mean_riemann(target_rct_id)))) for ti in target_rct_id])  # move target to id

    # subsampling 
    idx_s, idx_t = find_subsample_idx(source_rct_id, target_rct_id)

    axs2.append(fig2.add_subplot(1, 5, 3))
    axs2[2].set_title("Stretch at id")
    plot_diffusion_embedding(source_rct_id, target_rct_id, axs2[2], idx_s, idx_t)

    target_subsample_id = target_rct_id[idx_t]
    source_subsample_id = source_rct_id[idx_s]

    # find nns in subsamples
    nns, _ = find_nearest_neighbors(source_subsample_id, target_subsample_id, 'riem')
    target_subsample_id = make_pairs(source_subsample_id, target_subsample_id, nns) # rearrange target acc to nns

    ### find rotation with subsamples only ###
    R = get_rotation_matrix(M=source_subsample_id, Mtilde=target_subsample_id, weights=None, dist='rie')
    #print(R[1])
    R=R[0] # tuple due to verboselevel=2 in solve()
    print("Rotation found: ", R)
    target_rt_id = np.stack([np.dot(R, np.dot(t, R.T)) for t in target_rct_id]) # apply rotation to all samples

    axs2.append(fig2.add_subplot(1, 5, 4))
    axs2[3].set_title("Rotate wrt subsamples")
    plot_diffusion_embedding(source_rct_id, target_rt_id, axs2[3], idx_s, idx_t)

    ### move to source ###
    target = np.stack([np.dot(sqrtm(mean_source), np.dot(ti, sqrtm(mean_source))) for ti in target_rt_id]) 

    axs2.append(fig2.add_subplot(1, 5, 5))
    axs2[4].set_title("Move to source")
    plot_diffusion_embedding(source, target, axs2[4], idx_s, idx_t)
    fig2.savefig(results_path+"/icp_initial_iteration.pdf", bbox_inches='tight')

    return target


def recenter_to_id_stretch_recenter_to_source_rotate(source, target, calculate_stretching_factor=True, s=1):
    mean_target = mean_riemann(target)
    mean_source = mean_riemann(source)
    target_rct_id = np.stack(
        [np.dot(invsqrtm(mean_target), np.dot(ti, invsqrtm(mean_target))) for ti in target])  # move target to id

    ### stretch target at id ###
    if calculate_stretching_factor:
        source_rct_id = np.stack([np.dot(invsqrtm(mean_source), np.dot(si, invsqrtm(mean_source))) for si in source])  # move source to id
    
        disp_source = np.sum([distance_riemann(covi, np.eye(3)) ** 2 for covi in source_rct_id]) / len(source_rct_id)  # get stretching factor
        disp_target = np.sum([distance_riemann(covi, np.eye(3)) ** 2 for covi in target_rct_id]) / len(target_rct_id)
        s = round(np.sqrt(disp_target / disp_source),4)
        target_rct_id = np.stack([powm(ti, 1.0 / s) for ti in target_rct_id])  # stretch target at id

    target_ctr_source_mean = np.stack([np.dot(sqrtm(mean_source), np.dot(ti, sqrtm(mean_source))) for ti in target_rct_id])      # move target to source                                                           # move target to source mean
    target_rct_stretched = target_ctr_source_mean

    ### rotate ###
    source, target_rt, rotation_matrix_iter = rotate(source, target_rct_stretched)
    print("Calculated scaling factor: ", s)
    return target_rt, s, rotation_matrix_iter


def recenter_to_source_stretch_rotate(source, target, calculate_stretching_factor=True, s=1):
    mean_target = mean_riemann(target)
    mean_source = mean_riemann(source)
    target_rct_id = np.stack(
        [np.dot(invsqrtm(mean_target), np.dot(ti, invsqrtm(mean_target))) for ti in target])  # move target to id

    ### stretch target at source ###
    target_ctr_source_mean = np.stack([np.dot(sqrtm(mean_source), np.dot(ti, sqrtm(mean_source))) for ti in target_rct_id])      # move target to source 
    if calculate_stretching_factor:
        disp_source = np.sum([distance_riemann(covi, mean_source) ** 2 for covi in source]) / len(
            source)  # get stretching factor
        disp_target = np.sum([distance_riemann(covi, mean_source) ** 2 for covi in target_ctr_source_mean]) / len(
            target_ctr_source_mean)
        s = round(np.sqrt(disp_target / disp_source),4)
    target_rct_stretched = np.stack([powm(ti, 1.0 / s) for ti in target_ctr_source_mean])  # stretch target at source mean

    ### rotate ###
    source, target_rt, rotation_matrix_iter = rotate(source, target_rct_stretched)
    print("Calculated scaling factor: ", s)
    return target_rt, s, rotation_matrix_iter


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

    ### rotate ###
    target, rotation_matrix_iter = rotate(source, target)

    ### move to source ###
    target = np.stack([np.dot(sqrtm(mean_source_org), np.dot(ti, sqrtm(mean_source_org))) for ti in target]) 

    return target, s, rotation_matrix_iter


def rotate(source, target, weights=None, distance='rie'):  # rie or euc
    weights=np.ones(target.shape[0])
    R = get_rotation_matrix(M=source, Mtilde=target, weights=weights, dist=distance, x=np.eye(3))
    print(R[1])
    R=R[0] # tuple due to verboselevel=2 in solve()
    print(R)
    target = np.stack([np.dot(R, np.dot(t, R.T)) for t in target])
    #target_rotated = np.stack([np.dot(R.T, np.dot(t, R)) for t in target])
    return target, R


def perform_rotation(target, R):
    for r in R:
        target_rot = np.stack([np.dot(r, np.dot(t, r.T)) for t in target])
        #target_rot = np.stack([np.dot(r.T, np.dot(t, r)) for t in target])
    return target_rot


def perform_scaling(target, s):
    return np.stack([powm(covi, s) for covi in target])


def perform_translation_and_stretching(target, T, s):
    mean_target = mean_riemann(target)
    target_rct_id = np.stack([np.dot(invsqrtm(mean_target), np.dot(ti, invsqrtm(mean_target))) for ti in target])
    target_rct_id_stretched = target_rct_id
    for si in s:
        target_rct_id_stretched = np.stack([powm(ti, si) for ti in target_rct_id_stretched])  # stretch target
    target_rct_at_source_mean = np.stack([np.dot(sqrtm(T), np.dot(ti, sqrtm(T))) for ti in target_rct_id_stretched])
    return target_rct_at_source_mean

def perform_transformation(target, T, R, s):
    for i in np.arange(len(R)):
        # recenter to id
        mean_target = mean_riemann(target)
        target = np.stack([np.dot(invsqrtm(mean_target), np.dot(ti, invsqrtm(mean_target))) for ti in target])
        # stretch
        target = np.stack([powm(covi, s[i]) for covi in target])
        # rotate
        target = np.stack([np.dot(R[i], np.dot(t, R[i].T)) for t in target])
        #target = np.stack([np.dot(R[i].T, np.dot(t, R[i])) for t in target])
        # recenter to cource
        target = np.stack([np.dot(sqrtm(T[i]), np.dot(ti, sqrtm(T[i]))) for ti in target])
    return target



parser = argparse.ArgumentParser()
parser.add_argument("base_path", help="base_path.", type=str)
parser.add_argument("robot_teacher", help="robot_teacher.", type=str)
parser.add_argument("robot_student", help="robot_student", type=str)
parser.add_argument("lookup_dataset", help="path to lookup dataset e.g. 5000", type=str)
parser.add_argument("map_dataset", help="path to data to be mapped", type=str)
parser.add_argument("map_run", help="0: only find mapping params, 1: only new points, 2: both, 3: only new points no plot", type=int)

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

iter = 2

source = np.genfromtxt(args.base_path+"/"+args.robot_student+"/"+args.lookup_dataset+"/manipulabilities.csv", delimiter=',')
target = np.genfromtxt(args.base_path+"/"+args.robot_teacher+"/"+args.lookup_dataset+"/manipulabilities.csv", delimiter=',')

source_org = source.reshape((source.shape[0], 3, 3))
target = target.reshape((target.shape[0], 3, 3))
#target = copy.deepcopy(target_org)


### the alignment is performed s.t. target matches source afterwards, so target is rotated, source not
if run_map_find:
    results_path = args.base_path+"/"+args.robot_teacher+"/"+args.lookup_dataset

    R = np.zeros((iter, 3, 3))
    T = np.zeros((iter, 3, 3))
    s = np.zeros(iter)


    fig2 = plt.figure(figsize=(20, 7))
    fig2.suptitle('Diffusion Map')
    axs2 = list()
    axs2.append(fig2.add_subplot(1, iter + 2, 1))
    axs2[0].set_title("Original")
    plot_diffusion_embedding(source_org, target, axs2[0])

    target = initial_iteration(source_org, target, args.base_path+"/"+args.robot_student+"/"+args.lookup_dataset)

    ### find nns ###
    nns, _ = find_nearest_neighbors(source_org, target, 'riem')
    target = make_pairs(source_org, target, nns) # rearrange target acc to nns

    axs2.append(fig2.add_subplot(1, iter + 2, 2))
    axs2[1].set_title("Initial (%.2f)" %(get_mean_dist_nns(source_org, target)))
    plot_diffusion_embedding(source_org, target, axs2[1])

    print("-------------------------------------------------------------")

    for iter_i in tqdm(np.arange(iter)):
        print("Iteration %i/%i" %(iter_i, iter))

        #target_rt, s_iter, rotation_matrix_iter = recenter_to_source_stretch_rotate(source, target)
        #target_rt, s_iter, rotation_matrix_iter = recenter_to_id_stretch_recenter_to_source_rotate(source, target)
        target, s_iter, rotation_matrix_iter = recenter_to_id_stretch_rotate_recenter_to_source(source_org, target)

        nns, mean_dist_nns = find_nearest_neighbors(source_org, target, 'riem')
        target = make_pairs(source_org, target, nns)
    
        R[iter_i] = rotation_matrix_iter
        s[iter_i] = s_iter
        T[iter_i] = mean_riemann(source_org)

        print("R: ", R[iter_i])
        print("s: ", s[iter_i])
        print("T: ", T[iter_i])
        #print("mean riemann source/target: \n", mean_riemann(source_org), "\n", mean_riemann(target))

        axs2.append(fig2.add_subplot(1, iter + 2, iter_i + 3))
        plot_diffusion_embedding(source_org, target, axs2[iter_i + 2])
        axs2[iter_i + 2].set_title("Iteration %i - %.2f" % (iter_i, mean_dist_nns))

        print("-------------------------------------------------------------")

    # save results
    np.savetxt(results_path+"/R_icp_"+args.robot_teacher+"_to_"+args.robot_student+".txt", np.reshape(R, (iter, 9)), delimiter=',')
    np.savetxt(results_path+"/T_icp_"+args.robot_teacher+"_to_"+args.robot_student+".txt", np.reshape(T, (iter, 9)), delimiter=',')
    np.savetxt(results_path+"/s_icp_"+args.robot_teacher+"_to_"+args.robot_student+".txt", s, delimiter=',')

    fig2.tight_layout()
    results_path = args.base_path+"/"+args.robot_student+"/"+args.lookup_dataset
    fig2.savefig(results_path+"/icp_diffusion_map.pdf", bbox_inches='tight')



# apply trafo to new points
if run_map_new:
    results_path = args.base_path+"/"+args.robot_teacher+"/"+args.lookup_dataset
    target_new = np.genfromtxt(args.base_path+"/"+args.robot_teacher+"/"+args.map_dataset+"/manipulabilities_interpolated.csv", delimiter=',')

    R = np.genfromtxt(results_path+"/R_icp_"+args.robot_teacher+"_to_"+args.robot_student+".txt", delimiter=',')
    T = np.genfromtxt(results_path+"/T_icp_"+args.robot_teacher+"_to_"+args.robot_student+".txt", delimiter=',')
    s = np.genfromtxt(results_path+"/s_icp_"+args.robot_teacher+"_to_"+args.robot_student+".txt", delimiter=',')
    R = np.reshape(R, (iter, 3, 3))
    T = np.reshape(T, (iter, 3, 3))
    s=s.ravel()
    print(s)
    print(T)
    print(R)

    target_new = target_new[1:,1:].reshape((target_new[1:,1:].shape[0], 3, 3))

    fig2 = plt.figure(figsize=(16.5, 6))
    axs2 = list()
    axs2.append(fig2.add_subplot(1, 2, 1))
    axs2[0].set_title("Diffusion")
    plot_diffusion_embedding_target_new(source_org, target, target_new, axs2[0])

    print(mean_riemann(source_org))
    target_new = perform_transformation(target_new, T, R, s)

    np.savetxt(args.base_path+"/"+args.robot_student+"/"+args.map_dataset+"/manipulabilities_interpolated_mapped_icp.csv", np.reshape(target_new, (target_new.shape[0], 9)), delimiter=",")
    axs2.append(fig2.add_subplot(1, 2, 2))


    #plot naive mapping too in 2d embedding to be able to compare results
    target_mapped_naive = np.genfromtxt(args.base_path+"/"+args.robot_student+"/"+args.map_dataset+"/manipulabilities_interpolated_mapped_naive.csv", delimiter=',', dtype=np.double)
    target_mapped_naive = target_mapped_naive.reshape((target_mapped_naive.shape[0], 3, 3)) # + 1e-3
    for i in target:
        assert(np.all(np.linalg.eigvals(i) > 0))
    for i in target_new:
        assert(np.all(np.linalg.eigvals(i) > 0))
    for i in target_mapped_naive:
        assert(np.all(np.linalg.eigvals(i) > 0))

    plot_diffusion_embedding_target_new_and_naive(source_org, target, target_new, target_mapped_naive, axs2[1])

    fig2.tight_layout()
    results_path = args.base_path+"/"+args.robot_student+"/"+args.map_dataset
    fig2.savefig(results_path+"/mapped_diffusion_map.pdf")


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
