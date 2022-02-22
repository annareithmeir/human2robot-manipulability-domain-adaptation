import numpy as np
import matplotlib.pyplot as plt
from pyriemann.utils.distance import distance_riemann, distance_wasserstein
from SPDMetrics import distance_logeuc
from pyriemann.utils.base import invsqrtm, sqrtm, logm, expm, powm
from rpa.helpers.transfer_learning.utils import mean_riemann, parallel_transport_covariances
from get_rotation_matrix import get_rotation_matrix
#from rpa.helpers.transfer_learning.manopt import get_rotation_matrix
from tqdm import tqdm
from numpy import linalg as LA
from plot_2d_embeddings import plot_diffusion_embedding, \
    plot_diffusion_embedding_target_new, plot_diffusion_embedding_target_new_and_naive
import tkinter
import copy
from scipy.linalg import eigh
import matplotlib
import math
from path_sing2sing import find_singular_geodesic_paths, find_pairs_conv, find_most_singular_points, find_most_singular_points_conv, find_most_singular_points_diff_dir

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import argparse

#np.random.seed(0)
matplotlib.use('TkAgg')

def gen_symm(n):
    A = np.random.randn(n,n)
    return A + A.T

def gen_orth(n):
    A = gen_symm(n)
    _,Q = eigh(A)
    return Q

'''
Based on the following paper: Riemannian Procrustes Analysis
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8588384
'''


def make_pairs(source, target, nns):
    target_rearranged = np.zeros((source.shape[0],3,3))

    for i in np.arange(target_rearranged.shape[0]):
        target_rearranged[i] = target[nns[i]]

    return target_rearranged


def assign_distance_weights(source, target, w):
    ww=np.zeros(source.shape[0])
    for i in np.arange(source.shape[0]):
        ww[i]=w[i]*1/(distance_riemann(source[i], target[i])**2+0.001) # 0.001 to avoid division through 0
    return ww


# find nns with removing duplicates
# nns[i] is idx for which target_idx is closest to source_i 
def find_nearest_neighbors(source, target, distance='rie', filterdup=True):
    nns = np.zeros(source.shape[0], dtype=np.int)  # for each target search smallest source
    w=np.ones(source.shape[0], dtype=np.int)
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
    #print(nns)
    
    if filterdup:
        # remove duplicates
        #print("Found %i duplicates and %i unique NN pairs" %(np.argwhere(duplicates>1).shape[0], np.unique(nns).shape[0]))

        for i in np.arange(duplicates.shape[0]):
            if duplicates[i] > 1:
                stmp=np.argwhere(nns==i).ravel()
                d=np.zeros(stmp.shape[0])
                for s_idx in np.arange(stmp.shape[0]):
                    if distance == 'rie':
                        d[s_idx] = distance_riemann(source[stmp[s_idx]], target[i])
                    elif distance == 'fro':
                        d[s_idx] = LA.norm(logm(invsqrtm(source[stmp[s_idx]]) * target[i] * invsqrtm(source[stmp[s_idx]])), 'fro')
                    elif distance == 'was':
                        d[s_idx] = distance_wasserstein(source[stmp[s_idx]], target[i])
                    else:
                        print("[ERROR] either rie, was or fro!")
                        return nns

                min_idx= np.argmin(d)
                stmp = np.delete(stmp, min_idx)
                #nns[stmp] = -1
                w[stmp]=0

        target = make_pairs(source, target, nns)

        mean_dist_2 = 0.0
        for (s,t, wi) in zip(source, target, w):
            if distance == 'rie':
                mean_dist_2+= wi*distance_riemann(t, s)
            elif distance == 'fro':
                mean_dist_2+= wi*LA.norm(logm(invsqrtm(t) * s * invsqrtm(t)), 'fro')
            elif distance == 'was':
                mean_dist_2+= wi*distance_wasserstein(t,s)
            else:
                print("[ERROR] either rie, was or fro!")
                return nns
        mean_dist_2 = mean_dist_2/target.shape[0]


        w = reject_median(source, target, w, distance)

    else:
        target = make_pairs(source, target, nns)
        mean_dist_2 = -1

    mean_dist_3 = 0.0
    for (s,t, wi) in zip(source, target, w):
        if distance == 'rie':
           mean_dist_3+= wi*distance_riemann(t, s)
        elif distance == 'fro':
           mean_dist_3+= wi*LA.norm(logm(invsqrtm(t) * s * invsqrtm(t)), 'fro')
        elif distance == 'was':
           mean_dist_3+= wi*distance_wasserstein(t,s)
        else:
           print("[ERROR] either rie, was or fro!")
           return nns
    mean_dist_3 = mean_dist_3/target.shape[0]


    print("After rejection %i pairs left." %(np.sum(w)))
    print("Mean dists between nns found (before/after removal, reject median): %.3f, %.3f, %.3f" %  (mean_dist, mean_dist_2, mean_dist_3))

    return source, target, w, mean_dist_3

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


def reject_median(source,target, w, dist='rie'):
    distances=np.zeros(source.shape[0])
    if dist=='rie':
        for i in np.arange(source.shape[0]):
            distances[i] = distance_riemann(source[i], target[i])
    if dist=='was':
        for i in np.arange(source.shape[0]):
            distances[i] = distance_wasserstein(source[i], target[i])
    idx = np.argsort(distances) # indices from small to large
    #print(idx)
    #print(idx[int(idx.shape[0]/2):])
    w[idx[int(idx.shape[0]/2):]] = 0 # only keep lower half
    return w


def get_mean_dist_pairs(source, target, distance='rie'):
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
        dst_sum += ds*ds
    mean_dist = float(math.sqrt(dst_sum / target.shape[0]))
    #print("Mean RMSE between source and target pairs given: ", mean_dist)
    return mean_dist


# for initial step we subsample the most farthest points from mean
def find_subsample_idx(source, target):
    #num_points = math.ceil(target.shape[0]*0.1)
    if(target.shape[0]>=50):
        num_points=6
    if(target.shape[0]<=20):
        num_points=3
    print("For initial NNS %i points are used" %(num_points))
    idx_t = (np.array([distance_riemann(covi, np.eye(3)) ** 2 for covi in target])).argsort()[-num_points:][::-1] # find num_points points with biggest distance to id
    idx_s = (np.array([distance_riemann(covi, np.eye(3)) ** 2 for covi in source])).argsort()[-num_points:][::-1] # find num_points points with biggest distance to id
    return idx_s, idx_t


# move to id, scale, then find nns then rotate
def initial_iteration(source_org, target, results_path, dist):
    print("------------------Initial Iteration-----------------------")
    target_org = copy.deepcopy(target) # if maxiter reached we need to use original target

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
    target = np.stack([np.dot(invsqrtm(mean_riemann(target)), np.dot(ti, invsqrtm(mean_riemann(target)))) for ti in target])  # move target to id

    # subsampling 
    #idx_s, idx_t = find_subsample_idx(source, target)

    # sing2sing
    source_s2s, target_s2s, idx_s, idx_t = find_singular_geodesic_paths(source, target, 2)
    assert(source_s2s.shape[0]==target_s2s.shape[0])

    axs2.append(fig2.add_subplot(1, 5, 3))
    axs2[2].set_title("Stretch at id")
    plot_diffusion_embedding(source, target, axs2[2], idx_s, idx_t)
    #plot_diffusion_embedding(source, target, axs2[2])

    #target_subsample_id = target[idx_t]
    #source_subsample_id = source[idx_s]

    # find nns in subsamples
    #source_subsample_id, target_subsample_id, _ = find_nearest_neighbors(source_subsample_id, target_subsample_id, dist)


    ### find rotation with subsamples only ###
    


    itermax=100
    iter_curr=0
    mean_t=999
    mean_all=999
    R_all=list()
    err_all=list()
    while (iter_curr < itermax) and (mean_t > 1e-1*5):
        #source_nns, target_nns, _ = find_nearest_neighbors(source, target, dist)

        R = get_rotation_matrix(M=source, Mtilde=target, weights=None, dist='rie', x=gen_orth(3))
        #R = get_rotation_matrix(M=source_subsample_id, Mtilde=target_subsample_id, weights=None, dist='rie', x=gen_orth(3))
        #R = get_rotation_matrix(M=source, Mtilde=target, weights=None, dist='rie', x=gen_orth(3))
        #R = get_rotation_matrix(M=source_nns, Mtilde=target_nns, weights=None, dist='rie', x=gen_orth(3))
        #print(R[1])
        #R=R[0] # tuple due to verboselevel=2 in solve()
        #print("\nRotation found: \n", R)

        # apply rotation to all samples
        #target_s2s = np.stack([np.dot(R, np.dot(t, R.T)) for t in target_s2s]) 
        #target_nns = np.stack([np.dot(R, np.dot(t, R.T)) for t in target_nns]) 
        #target_subsample_id = np.stack([np.dot(R, np.dot(t, R.T)) for t in target_subsample_id]) 
        #target = np.stack([np.dot(R, np.dot(t, R.T)) for t in target]) 
        #R_all.append(R)

        #####
        #print("Distances after iteration %i" %(iter_curr))
        mean_t=0
        for i in np.arange(source.shape[0]):
            #print(distance_riemann(source_nns[i], target_nns[i]))
            #mean_t+= distance_riemann(source_nns[i], target_nns[i])
            mean_all+= distance_riemann(source[i], target[i])

        for i in np.arange(source_s2s.shape[0]):
            mean_t+= distance_riemann(source_s2s[i], target_s2s[i])

        #####

        iter_curr+=1
        mean_t =mean_t/target_s2s.shape[0]
        mean_all =mean_all/target.shape[0]
        #err_all.append(mean_all)
        err_all.append(mean_t)
        print("Mean Distance after iteration %i: %.3f // %.3f" %(iter_curr, mean_t, mean_all))
    print("Mean Distance reached after iteration %i" %(iter_curr))
    R_all.append(R)
    target = np.stack([np.dot(R, np.dot(t, R.T)) for t in target]) 

    # if(iter_curr == itermax):
    #     print("[MAX ITER] Looking for smallest error because maxiter reached!")
    #     min_idx = err_all.index(min(err_all))
    #     print("best iteration found: %i" %(min_idx))
    #     R_all=R_all[:min_idx]
    #     target=target_org

    #     for r in R_all:
    #         target = np.stack([np.dot(r, np.dot(t, r.T)) for t in target]) 

    #     mean_t=0
    #     for i in np.arange(target.shape[0]):
    #         mean_t+= distance_riemann(source[i], target[i])
    #     print("New distance after selecting best: %.3f" %(mean_t/target.shape[0]))
    #     #####

    axs2.append(fig2.add_subplot(1, 5, 4))
    axs2[3].set_title("Rotate wrt subsamples")
    #plot_diffusion_embedding(source, target, axs2[3], idx_s, idx_t)
    plot_diffusion_embedding(source, target, axs2[3])


    ### move to source ###
    target = np.stack([np.dot(sqrtm(mean_source), np.dot(ti, sqrtm(mean_source))) for ti in target]) 

    axs2.append(fig2.add_subplot(1, 5, 5))
    axs2[4].set_title("Move to source")
    plot_diffusion_embedding(source_org, target, axs2[4], idx_s, idx_t)
    #plot_diffusion_embedding(source_org, target, axs2[4])
    fig2.savefig(results_path+"/icp_initial_iteration.pdf", bbox_inches='tight')

    print("-------------------------------------------------------------")

    return R_all, np.ravel(s), np.ravel(mean_source), target


# move to id, scale, then find nns then rotate
def initial_iteration_subsamples(source_org, target, results_path, dist):
    print("------------------Initial Iteration-----------------------")
    target_org = copy.deepcopy(target) # if maxiter reached we need to use original target

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
    target = np.stack([np.dot(invsqrtm(mean_riemann(target)), np.dot(ti, invsqrtm(mean_riemann(target)))) for ti in target])  # move target to id

    # subsampling 
    idx_s, idx_t = find_subsample_idx(source, target)

    axs2.append(fig2.add_subplot(1, 5, 3))
    axs2[2].set_title("Stretch at id")
    plot_diffusion_embedding(source, target, axs2[2], idx_s, idx_t)

    target_subsample_id = target[idx_t]
    source_subsample_id = source[idx_s]

    # find nns in subsamples
    source_subsample_id, target_subsample_id, w, _ = find_nearest_neighbors(source_subsample_id, target_subsample_id, dist)


    ### find rotation with subsamples only ###
    itermax=100
    iter_curr=0
    mean_t=999
    mean_all=999
    R_all=list()
    err_all=list()
    while (iter_curr < itermax) and (mean_t > 1e-1*5):
        #source_nns, target_nns, _ = find_nearest_neighbors(source, target, dist)
        R = get_rotation_matrix(M=source_subsample_id, Mtilde=target_subsample_id, weights=w, dist='rie', x=gen_orth(3))

        # apply rotation to all samples
        target_subsample_id = np.stack([np.dot(R, np.dot(t, R.T)) for t in target_subsample_id]) 
        target = np.stack([np.dot(R, np.dot(t, R.T)) for t in target]) 
        R_all.append(R)

        mean_t=0
        for i in np.arange(source.shape[0]):
            mean_all+= distance_riemann(source[i], target[i])

        for i in np.arange(source_subsample_id.shape[0]):
            mean_t+= distance_riemann(source_subsample_id[i], target_subsample_id[i])


        iter_curr+=1
        mean_t =mean_t/target_subsample_id.shape[0]
        mean_all =mean_all/target.shape[0]
        #err_all.append(mean_all)
        err_all.append(mean_t)
        print("Mean Distance after iteration %i: %.3f // %.3f" %(iter_curr, mean_t, mean_all))
    print("Mean Distance reached after iteration %i" %(iter_curr))


    if(iter_curr == itermax):
        print("[MAX ITER] Looking for smallest error because maxiter reached!")
        min_idx = err_all.index(min(err_all))
        print("best iteration found: %i" %(min_idx))
        R_all=R_all[:min_idx]
        target=target_org

        for r in R_all:
            target = np.stack([np.dot(r, np.dot(t, r.T)) for t in target]) 

        mean_t=0
        for i in np.arange(target.shape[0]):
            mean_t+= distance_riemann(source[i], target[i])
        print("New distance after selecting best: %.3f" %(mean_t/target.shape[0]))
        #####

    axs2.append(fig2.add_subplot(1, 5, 4))
    axs2[3].set_title("Rotate wrt subsamples")
    plot_diffusion_embedding(source, target, axs2[3], idx_s, idx_t)


    ### move to source ###
    target = np.stack([np.dot(sqrtm(mean_source), np.dot(ti, sqrtm(mean_source))) for ti in target]) 

    axs2.append(fig2.add_subplot(1, 5, 5))
    axs2[4].set_title("Move to source")
    plot_diffusion_embedding(source_org, target, axs2[4], idx_s, idx_t)
    fig2.savefig(results_path+"/icp_initial_iteration.pdf", bbox_inches='tight')

    print("-------------------------------------------------------------")

    return R_all, np.ravel(s), np.ravel(mean_source), target


# move to id, scale, then find nns then rotate
def initial_iteration_nns(source_org, target, results_path, dist):
    print("------------------Initial Iteration-----------------------")
    target_org = copy.deepcopy(target) # if maxiter reached we need to use original target

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
    target = np.stack([np.dot(invsqrtm(mean_riemann(target)), np.dot(ti, invsqrtm(mean_riemann(target)))) for ti in target])  # move target to id


    axs2.append(fig2.add_subplot(1, 5, 3))
    axs2[2].set_title("Stretch at id")
    plot_diffusion_embedding(source, target, axs2[2])
    ### find rotation with subsamples only ###
    


    itermax=100
    iter_curr=0
    mean_t=999
    mean_all=999
    R_all=list()
    err_all=list()
    while (iter_curr < itermax) and (mean_t > 1e-1*5):
        source_nns, target_nns, w, _ = find_nearest_neighbors(source, target, dist)
        R = get_rotation_matrix(M=source_nns, Mtilde=target_nns, weights=w, dist='rie', x=gen_orth(3))

        # apply rotation to all samples
        target = np.stack([np.dot(R, np.dot(t, R.T)) for t in target]) 
        R_all.append(R)

        mean_t=0
        for i in np.arange(source.shape[0]):
            mean_all+= distance_riemann(source[i], target[i])

        for i in np.arange(source_nns.shape[0]):
            mean_t+= distance_riemann(source_nns[i], target_nns[i])

        iter_curr+=1
        mean_t =mean_t/target_s2s.shape[0]
        mean_all =mean_all/target.shape[0]
        err_all.append(mean_t)
        print("Mean Distance after iteration %i: %.3f // %.3f" %(iter_curr, mean_t, mean_all))
    print("Mean Distance reached after iteration %i" %(iter_curr))

    if(iter_curr == itermax):
        print("[MAX ITER] Looking for smallest error because maxiter reached!")
        min_idx = err_all.index(min(err_all))
        print("best iteration found: %i" %(min_idx))
        R_all=R_all[:min_idx]
        target=target_org

        for r in R_all:
            target = np.stack([np.dot(r, np.dot(t, r.T)) for t in target]) 

        mean_t=0
        for i in np.arange(target.shape[0]):
            mean_t+= distance_riemann(source[i], target[i])
        print("New distance after selecting best: %.3f" %(mean_t/target.shape[0]))

    axs2.append(fig2.add_subplot(1, 5, 4))
    axs2[3].set_title("Rotate wrt subsamples")
    plot_diffusion_embedding(source, target, axs2[3], idx_s, idx_t)
    #plot_diffusion_embedding(source, target, axs2[3])


    ### move to source ###
    target = np.stack([np.dot(sqrtm(mean_source), np.dot(ti, sqrtm(mean_source))) for ti in target]) 

    axs2.append(fig2.add_subplot(1, 5, 5))
    axs2[4].set_title("Move to source")
    plot_diffusion_embedding(source_org, target, axs2[4], idx_s, idx_t)
    #plot_diffusion_embedding(source_org, target, axs2[4])
    fig2.savefig(results_path+"/icp_initial_iteration.pdf", bbox_inches='tight')

    print("-------------------------------------------------------------")

    return R_all, np.ravel(s), np.ravel(mean_source), target

# as in eq. (33) for one subsample
def initial_R_estimate(source_subsample, target_subsample):
    ws,vs =np.linalg.eigh(source_subsample)
    wt,vt =np.linalg.eigh(target_subsample)
    return np.dot(vt, vs.transpose())



# move to id, scale, then find nns then rotate
def initial_iteration_sing2sing(source_org, target, results_path, dist):


    ### find R and rotate all targets ###
    itermax=100
    iter_curr=0
    mean_t=999
    mean_nns=999
    R=list()
    err_all=list()

    # sing2sing
    source_s2s, target_s2s, idx_s, idx_t = find_singular_geodesic_paths(source, target, 3)
    assert(source_s2s.shape[0]==target_s2s.shape[0])
    print("SING2SING using %i samples now" %(source_s2s.shape[0]))

    while (iter_curr < itermax) and (mean_nns > 1e-1*1):
        w=np.ones(target.shape[0])
        # w=np.ones(target_s2s.shape[0])
        #w=assign_distance_weights(source_s2s, target_s2s, w)
        rotation_matrix_iter = get_rotation_matrix(M=source_nns, Mtilde=target_nns, weights=None, dist='rie', x=gen_orth(3))
        # target_tmp = np.stack([np.dot(rotation_matrix_iter, np.dot(t, rotation_matrix_iter.T)) for t in target])
        target_tmp = np.stack([np.dot(rotation_matrix_iter, np.dot(t, rotation_matrix_iter.T)) for t in target_nns])

        mean_t=0
        mean_nns=0
        for i in np.arange(source_nns.shape[0]):
            # mean_t+= distance_riemann(source[i], target[i])
            mean_nns+= distance_riemann(source_nns[i], target_tmp[i])
        iter_curr+=1
        # mean_t =mean_t/target.shape[0]
        mean_nns =mean_nns/target_nns.shape[0]
        # mean_nns =mean_nns/target_nns.shape[0]
        print("Mean after iteration %i: %.3f" %(iter_curr, mean_nns))
        err_all.append(mean_nns)
        R.append(rotation_matrix_iter)

    if(iter_curr == itermax):
        print("[MAX ITER] Looking for smallest error because maxiter reached!")
        min_idx = err_all.index(min(err_all))
        print("best iteration found: %i" %(min_idx))
        rotation_matrix_iter=R[min_idx]
        
    else:
        rotation_matrix_iter = R[-1]

    target = np.stack([np.dot(rotation_matrix_iter, np.dot(t, rotation_matrix_iter.T)) for t in target])

    return target, s, [rotation_matrix_iter]


def icp_iteration_sing2sing(source_org, target, dist):

    ### find R and rotate all targets ###
    itermax=100
    iter_curr=0
    mean_t=999
    mean_nns=999
    R=list()
    err_all=list()

    source_nns, target_nns, idx_s, idx_t, w = find_singular_geodesic_paths(source, target, 6) # num points to use
    print("IDX LEN: ", idx_s.shape, idx_t.shape, w.shape)
    while (iter_curr < itermax) and (mean_nns > 1e-1*1):
        rotation_matrix_iter = get_rotation_matrix(M=source_nns, Mtilde=target_nns, weights=w, dist='rie', x=gen_orth(3))
        # target_tmp = np.stack([np.dot(rotation_matrix_iter, np.dot(t, rotation_matrix_iter.T)) for t in target])
        target_tmp = np.stack([np.dot(rotation_matrix_iter, np.dot(t, rotation_matrix_iter.T)) for t in target_nns])

        mean_t=0
        mean_nns=0
        for i in np.arange(source_nns.shape[0]):
            # mean_t+= distance_riemann(source[i], target[i])
            mean_nns+= distance_riemann(source_nns[i], target_tmp[i])
        iter_curr+=1
        # mean_t =mean_t/target.shape[0]
        mean_nns =mean_nns/target_nns.shape[0]
        # mean_nns =mean_nns/target_nns.shape[0]
        print("Mean after iteration %i: %.3f" %(iter_curr, mean_nns))
        err_all.append(mean_nns)
        R.append(rotation_matrix_iter)

    if(iter_curr == itermax):
        print("[MAX ITER] Looking for smallest error because maxiter reached!")
        min_idx = err_all.index(min(err_all))
        print("best iteration found: %i" %(min_idx))
        rotation_matrix_iter=R[min_idx]
        
    else:
        rotation_matrix_iter = R[-1]

    target = np.stack([np.dot(rotation_matrix_iter, np.dot(t, rotation_matrix_iter.T)) for t in target])

    return target, s, [rotation_matrix_iter], idx_s, idx_t


def icp_iteration_most_singular(source_org, target, dist):

    ### find R and rotate all targets ###
    itermax=10
    iter_curr=0
    mean_t=999
    mean_nns=999
    R=list()
    err_all=list()
    err_iter=0


    #source_nns, target_nns, idx_s, idx_t = find_most_singular_points_diff_dir(source_org, target, 6)
    #source_nns, target_nns, idx_s, idx_t, w = find_most_singular_points_conv(source_org, target, 25) # 12 best so far, 75 for human to robot
    source_nns, target_nns, idx_s, idx_t, w = find_pairs_conv(source_org, target)
    #source_nns, target_nns, idx_s, idx_t, w = find_most_singular_points(source_org, target, 12) # 12 best so far
    print("USING %i MOST SINGULAR POINTS " %(idx_s.shape[0]))
    #r_init_guess = initial_R_estimate(source_nns[np.argmax(w)], target[np.argmax(w)])


    while (iter_curr < itermax) and (mean_nns > 1e-1*1):
        rotation_matrix_iter = get_rotation_matrix(M=source_nns, Mtilde=target_nns, weights=w, dist='rie', x=gen_orth(3)) # www best for robot human
        target_tmp = np.stack([np.dot(rotation_matrix_iter, np.dot(t, rotation_matrix_iter.T)) for t in target_nns])

        mean_t=0
        mean_nns=0
        for i in np.arange(source_nns.shape[0]):
            mean_nns+= distance_riemann(source_nns[i], target_tmp[i])
        iter_curr+=1
        mean_nns = mean_nns/target_nns.shape[0]
        print("Mean after iteration %i: %.3f" %(iter_curr, mean_nns))
        err_all.append(mean_nns)
        R.append(rotation_matrix_iter)

    if(iter_curr == itermax):
        print("[MAX ITER] Looking for smallest error because maxiter reached!")
        min_idx = err_all.index(min(err_all))
        print("best iteration found: %i with MSE: %.3f" %(min_idx, err_all[min_idx]))
        rotation_matrix_iter=R[min_idx]
        err_iter = err_all[min_idx]
        
    else:
        rotation_matrix_iter = R[-1]
        err_iter = err_all[-1]

    target = np.stack([np.dot(rotation_matrix_iter, np.dot(t, rotation_matrix_iter.T)) for t in target])

    return target, s, [rotation_matrix_iter], idx_s, idx_t, err_iter


def icp_iteration(source_org, target, dist):

    ### find R and rotate all targets ###
    itermax=100
    iter_curr=0
    mean_t=999
    mean_nns=999
    R=list()
    err_all=list()


    source_nns, target_nns, w, mean_dist_nns = find_nearest_neighbors(source, target, dist, filterdup=False)

    while (iter_curr < itermax) and (mean_nns > 1e-1*1):
        rotation_matrix_iter = get_rotation_matrix(M=source_nns, Mtilde=target_nns, weights=w, dist='rie', x=gen_orth(3))
        # target_tmp = np.stack([np.dot(rotation_matrix_iter, np.dot(t, rotation_matrix_iter.T)) for t in target])
        target_tmp = np.stack([np.dot(rotation_matrix_iter, np.dot(t, rotation_matrix_iter.T)) for t in target_nns])

        mean_t=0
        mean_nns=0
        for i in np.arange(source.shape[0]):
            # mean_t+= distance_riemann(source[i], target[i])
            mean_nns+= w[i]*distance_riemann(source_nns[i], target_tmp[i])
        iter_curr+=1
        # mean_t =mean_t/target.shape[0]
        mean_nns =mean_nns/np.sum(w)
        # mean_nns =mean_nns/target_nns.shape[0]
        print("Mean after iteration %i: %.3f" %(iter_curr, mean_nns))
        err_all.append(mean_nns)
        R.append(rotation_matrix_iter)

    if(iter_curr == itermax):
        print("[MAX ITER] Looking for smallest error because maxiter reached!")
        min_idx = err_all.index(min(err_all))
        print("best iteration found: %i" %(min_idx))
        rotation_matrix_iter=R[min_idx]
        
    else:
        rotation_matrix_iter = R[-1]

    target = np.stack([np.dot(rotation_matrix_iter, np.dot(t, rotation_matrix_iter.T)) for t in target])

    return target, s, [rotation_matrix_iter]


def icp_iteration_pairs(source_org, target, dist):

    ### find R and rotate all targets ###
    itermax=1
    iter_curr=0
    mean_t=999
    mean_nns=999
    R=list()
    err_all=list()
    err_iter=0

    while (iter_curr < itermax) and (mean_t > 1e-1*1):
        rotation_matrix_iter = get_rotation_matrix(M=source_org, Mtilde=target, weights=None, dist='rie', x=gen_orth(3))
        # target_tmp = np.stack([np.dot(rotation_matrix_iter, np.dot(t, rotation_matrix_iter.T)) for t in target])
        target_tmp = np.stack([np.dot(rotation_matrix_iter, np.dot(t, rotation_matrix_iter.T)) for t in target])

        mean_t=0
        for i in np.arange(source_org.shape[0]):
            mean_t+= distance_riemann(source_org[i], target[i])
        iter_curr+=1
        mean_t =mean_t/target.shape[0]
        print("Mean after iteration %i: %.3f" %(iter_curr, mean_t))
        err_all.append(mean_t)
        R.append(rotation_matrix_iter)

    if(iter_curr == itermax):
        print("[MAX ITER] Looking for smallest error because maxiter reached!")
        min_idx = err_all.index(min(err_all))
        print("best iteration found: %i" %(min_idx))
        rotation_matrix_iter=R[min_idx]
        err_iter=err_all[min_idx]
        
    else:
        rotation_matrix_iter = R[-1]
        err_iter=err_all[-1]

    target = np.stack([np.dot(rotation_matrix_iter, np.dot(t, rotation_matrix_iter.T)) for t in target])
    print("Mean distance between pairs= %.3f" %(get_mean_dist_pairs(source_org, target, dist)))

    return target, s, [rotation_matrix_iter], err_iter


def perform_transformation(source_org, target, T, R, s, map_dataset):
    fig3 = plt.figure(figsize=(20, 7))
    fig3.suptitle('\\textit{Mapping new data}')
    axs3 = list()

    axs3.append(fig3.add_subplot(1, 5, 1))
    axs3[0].set_title("\\textit{Original}")
    plot_diffusion_embedding(source_org, target, axs3[0])
    axs3[0].legend(loc='lower right')

    # recenter to id
    mean_target = mean_riemann(target)
    target = np.stack([np.dot(invsqrtm(mean_target), np.dot(ti, invsqrtm(mean_target))) for ti in target])

    axs3.append(fig3.add_subplot(1, 5, 2))
    axs3[1].set_title("\\textit{Recenter to id}")
    plot_diffusion_embedding(source, target, axs3[1])

    # stretch
    target = np.stack([powm(covi, s[0]) for covi in target])

    axs3.append(fig3.add_subplot(1, 5, 3))
    axs3[2].set_title("\\textit{Stretch at id}")
    plot_diffusion_embedding(source, target, axs3[2])

    for i in np.arange(len(R)):
        target = np.stack([np.dot(R[i], np.dot(t, R[i].T)) for t in target])


    axs3.append(fig3.add_subplot(1, 5, 4))
    axs3[3].set_title("\\textit{Rotate wrt subsamples}")
    plot_diffusion_embedding(source, target, axs3[3])

    # recenter to source (all Ti same)
    target = np.stack([np.dot(sqrtm(T[-1]), np.dot(ti, sqrtm(T[-1]))) for ti in target])


    axs3.append(fig3.add_subplot(1, 5, 5))
    axs3[4].set_title("\\textit{Move to source}")
    plot_diffusion_embedding(source_org, target, axs3[4])

    if not os.path.exists(final_results_path+"/validation/"+map_dataset):
        os.makedirs(final_results_path+"/validation/"+map_dataset)
    fig3.savefig(final_results_path+"/validation/"+map_dataset+"/mapping_process.pdf")

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

final_results_path="/home/nnrthmr/CLionProjects/ma_thesis/data/final_results/2dof-2dofscaled"
# final_results_path="/home/nnrthmr/CLionProjects/ma_thesis/data/final_results/sing_paths/ICP-NNconvw"

source = np.genfromtxt(args.base_path+"/"+args.robot_student+"/"+args.lookup_dataset+"/manipulabilities.csv", delimiter=',')
target = np.genfromtxt(args.base_path+"/"+args.robot_teacher+"/"+args.lookup_dataset+"/manipulabilities.csv", delimiter=',')

source_org = source.reshape((source.shape[0], 3, 3))
target_org = target.reshape((target.shape[0], 3, 3))

for i in np.arange(target_org.shape[0]):
    m=target_org[i]
    w,v = np.linalg.eigh(m)
    w[w<1e-12]=0.0001
    m=np.matmul(np.matmul(v, np.diag(w)), v.transpose())
    target_org[i]=m

for i in np.arange(source_org.shape[0]):
    m=source_org[i]
    w,v = np.linalg.eigh(m)
    w[w<1e-12]=0.0001
    m=np.matmul(np.matmul(v, np.diag(w)), v.transpose())
    source_org[i]=m

source_org = source_org[::10]
target_org = target_org[::10]

source = copy.deepcopy(source_org)
target = copy.deepcopy(target_org)

print("ORIGINAL RMSE GROUND TRUTH=%.3f"%(get_mean_dist_pairs(source, target)))

#shuf_order = np.arange(target.shape[0])
#np.random.shuffle(shuf_order)
#target = target[shuf_order]

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

    #R = np.zeros((iter, 3, 3))
    R_list=list()
    T = np.zeros((1, 3, 3))
    s = np.zeros(1)


    fig2 = plt.figure(figsize=(20, 7))
    fig2.suptitle('Diffusion Map')
    axs2 = list()
    axs2.append(fig2.add_subplot(1, iter + 1, 1))
    axs2[0].set_title("Original")
    plot_diffusion_embedding(source_org, target, axs2[0])
    axs2[0].legend(loc='lower right')

    print("-------------------------------------------------------------")

    fig3 = plt.figure(figsize=(20, 7))
    fig3.suptitle('\\textit{Initial iteration process}')
    axs3 = list()
    axs3.append(fig3.add_subplot(1, 5, 1))
    axs3[0].set_title("\\textit{Original}")
    plot_diffusion_embedding(source_org, target, axs3[0])
    axs3[0].legend(loc='lower right')

    


    mean_target = mean_riemann(target)
    mean_source_org = mean_riemann(source_org)
    print(target.shape)

    print("Original distance between means: %.3f" %(distance_riemann(mean_target, mean_source_org)))
    T[0]=mean_riemann(source_org)

    ### move target to id ###
    target = np.stack([np.dot(invsqrtm(mean_target), np.dot(ti, invsqrtm(mean_target))) for ti in target])  
    source = np.stack([np.dot(invsqrtm(mean_source_org), np.dot(si, invsqrtm(mean_source_org))) for si in source_org])  # move source to id

    axs3.append(fig3.add_subplot(1, 5, 2))
    axs3[1].set_title("\\textit{Recenter to id}")
    plot_diffusion_embedding(source, target, axs3[1])

    ### stretch target at id ###
    disp_source = np.sum([distance_riemann(covi, np.eye(3)) ** 2 for covi in source]) / len(source)  # get stretching factor
    disp_target = np.sum([distance_riemann(covi, np.eye(3)) ** 2 for covi in target]) / len(target)
    s[0] = 1.0 / round(np.sqrt(disp_target / disp_source),4)
    target = np.stack([powm(ti, s[0]) for ti in target])  # stretch target at id

    err_list=list()


    for iter_i in tqdm(np.arange(iter)):
        print("Iteration %i/%i" %(iter_i, iter))

        if iter_i == 0:
            target_tmpplot =copy.deepcopy(target)

        #target, s_iter, rotation_matrix_iter, idx_s, idx_t = icp_iteration_sing2sing(source, target, dist)
        target, s_iter, rotation_matrix_iter, idx_s, idx_t, err_iter = icp_iteration_most_singular(source, target, dist)
        # target, s_iter, rotation_matrix_iter, err_iter = icp_iteration_pairs(source, target, dist)
        assert(target.shape[0]==target_org.shape[0])
        for r in rotation_matrix_iter:
            R_list.append(r)

        if iter_i == 0: # just to get the indices into the plot
            axs3.append(fig3.add_subplot(1, 5, 3))
            axs3[2].set_title("\\textit{Stretch at id}")
            plot_diffusion_embedding(source, target_tmpplot, axs3[2])
            # plot_diffusion_embedding(source, target_tmpplot, axs3[2], idx_s, idx_t, True)

        axs2.append(fig2.add_subplot(1, iter + 1, iter_i + 2))
        plot_diffusion_embedding(source, target, axs2[iter_i + 1])
        # plot_diffusion_embedding(source, target, axs2[iter_i + 1], idx_s, idx_t)

        #source_tmp, target_tmp, _, _ = find_nearest_neighbors(source, target, dist, filterdup=False)
        # err_iter=get_mean_dist_pairs(source_tmp, target_tmp)
        #err_iter=get_mean_dist_pairs(source, target)
        err_list.append(err_iter)
        print("After iteration %i error between nn pairs is %.3f"%(iter_i, err_iter))
        axs2[iter_i + 1].set_title("%.2f" % (err_iter))

        # if(err_iter <=1):
        if(err_iter <=1e-3*8):
            break

    print("-------------------------------------------------------------")

    if iter == 0:
        R_list.append(np.eye(3))

    if iter==0:
        iter_i = 0
    if iter_i == (iter-1):
        print("[OUTER MAX ITER] Looking for smallest error because maxiter reached!")
        min_idx = err_list.index(min(err_list))
        print("best iteration found: %i with MSE: %.3f" %(min_idx, err_list[min_idx]))
        R_list = R_list[:min_idx+1]


    #axs3.append(fig3.add_subplot(1, 5, 4))
    #axs3[3].set_title("\\textit{Rotate wrt subsamples}")
    #plot_diffusion_embedding(source, target, axs3[3])

    # plot_diffusion_embedding(source, target, axs3[3], idx_s, idx_t, True)

    ### move to source ###
    target = np.stack([np.dot(sqrtm(mean_source_org), np.dot(ti, sqrtm(mean_source_org))) for ti in target]) 

    #axs3.append(fig3.add_subplot(1, 5, 5))
    #axs3[4].set_title("\\textit{Move to source}")
    #plot_diffusion_embedding(source_org, target, axs3[4])

    # plot_diffusion_embedding(source_org, target, axs3[4], idx_s, idx_t, True)
    fig3.savefig(results_path2+"/icp_initial_iteration.pdf", bbox_inches='tight')

    R_arr=np.zeros((len(R_list),3,3))
    for i in np.arange(len(R_list)):
        R_arr[i]=R_list[i]



    #unshuf_order = np.zeros_like(shuf_order)
    #unshuf_order[shuf_order] = np.arange(target.shape[0])
    #target = target[unshuf_order]

    print("AFTER ALL ITERATIONS MEAN DIST TO GROUND TRUTH: %.3f" % (get_mean_dist_pairs(source_org, target)) )


    # save results
    if iter==0:
        #np.savetxt(results_path+"/R_icp_"+args.robot_teacher+"_to_"+args.robot_student+".txt", np.reshape(R, (1, 9)), delimiter=',')
        np.savetxt(results_path+"/R_icp_"+args.robot_teacher+"_to_"+args.robot_student+".txt", np.reshape(R_arr, (len(R_list), 9)), delimiter=',')
        np.savetxt(results_path+"/T_icp_"+args.robot_teacher+"_to_"+args.robot_student+".txt", np.reshape(T, (1, 9)), delimiter=',')
        np.savetxt(results_path+"/s_icp_"+args.robot_teacher+"_to_"+args.robot_student+".txt", s, delimiter=',')
    else:
        #np.savetxt(results_path+"/R_icp_"+args.robot_teacher+"_to_"+args.robot_student+".txt", np.reshape(R, (iter, 9)), delimiter=',')
        np.savetxt(results_path+"/R_icp_"+args.robot_teacher+"_to_"+args.robot_student+".txt", np.reshape(R_arr, (len(R_list), 9)), delimiter=',')
        np.savetxt(results_path+"/T_icp_"+args.robot_teacher+"_to_"+args.robot_student+".txt", np.reshape(T, (1, 9)), delimiter=',')
        np.savetxt(results_path+"/s_icp_"+args.robot_teacher+"_to_"+args.robot_student+".txt", s, delimiter=',')

        np.savetxt(final_results_path+"/R_icp_"+args.robot_teacher+"_to_"+args.robot_student+".txt", np.reshape(R_arr, (len(R_list), 9)), delimiter=',')
        np.savetxt(final_results_path+"/T_icp_"+args.robot_teacher+"_to_"+args.robot_student+".txt", np.reshape(T, (1, 9)), delimiter=',')
        np.savetxt(final_results_path+"/s_icp_"+args.robot_teacher+"_to_"+args.robot_student+".txt", s, delimiter=',')
        with open(final_results_path+"/info.txt", 'w+') as f:
            f.write("Teacher: "+args.robot_teacher+"\nStudent: "+args.robot_student+"\nniter: "+str(iter)+"\nnpoints: "+str(source_org.shape[0]))
            if iter_i == (iter-1):
                f.write("\nBest iter: "+str(min_idx)+"\nBest RMSE: "+str(err_list[min_idx]))
                f.write("\nAfter RMSE to groundtruth: %.3f"%(get_mean_dist_pairs(source_org, target)))
        #fig2.savefig(final_results_path+"/icp_diffusion_map.pdf", bbox_inches='tight')
        fig3.savefig(final_results_path+"/icp_first_iteration.pdf", bbox_inches='tight')

    fig2.tight_layout()
    fig2.savefig(results_path2+"/icp.pdf", bbox_inches='tight')

print("-------------------------------------------------------------")
print("-------------------------------------------------------------")



# apply trafo to new points
if run_map_new:
    print("------> RUNNING NEW DATA")
    if args.cv_k is None:
        # target_new = np.genfromtxt(args.base_path+"/"+args.robot_teacher+"/"+args.map_dataset+"/manipulabilities_interpolated_normalized.csv", delimiter=',')
        # target_new = target_new[:,1:].reshape((target_new[:,1:].shape[0], 3, 3)) 
        # # target_new = target_new[1:,1:].reshape((target_new[1:,1:].shape[0], 3, 3))   
        # filename_manip_groundtruth=args.base_path+"/"+args.robot_student+"/"+args.map_dataset+"/manipulabilities_interpolated_groundtruth_normalized.csv" 
        # manip_groundtruth = np.genfromtxt(filename_manip_groundtruth, delimiter=',')
        # manip_groundtruth = manip_groundtruth[:,1:].reshape((manip_groundtruth[:,1:].shape[0], 3, 3))   

        if(args.robot_teacher=="rhuman"):
            # target_new = np.genfromtxt(args.base_path+"/"+args.robot_teacher+"/"+args.map_dataset+"/manipulabilities_20.csv", delimiter=',')
            target_new = np.genfromtxt(args.base_path+"/"+args.robot_teacher+"/"+args.map_dataset+"/manipulabilities_40.csv", delimiter=',')
            target_new = target_new.reshape((target_new.shape[0], 3, 3)) 
            # filename_manip_groundtruth=args.base_path+"/"+args.robot_student+"/"+args.map_dataset+"/manipulabilities_20.csv" 
            filename_manip_groundtruth=args.base_path+"/"+args.robot_student+"/"+args.map_dataset+"/manipulabilities_40.csv" 
            manip_groundtruth = np.genfromtxt(filename_manip_groundtruth, delimiter=',')
            manip_groundtruth = manip_groundtruth.reshape((manip_groundtruth.shape[0], 3, 3))   
            for i in np.arange(target_new.shape[0]):
                m=target_new[i]
                w,v = np.linalg.eigh(m)
                w[w<1e-12]=0.0001
                m=np.matmul(np.matmul(v, np.diag(w)), v.transpose())
                target_new[i]=m
            for i in np.arange(manip_groundtruth.shape[0]):
                m=manip_groundtruth[i]
                w,v = np.linalg.eigh(m)
                w[w<1e-12]=0.0001
                m=np.matmul(np.matmul(v, np.diag(w)), v.transpose())
                manip_groundtruth[i]=m

        elif(args.robot_teacher=="2dof"):
            # target_new = np.genfromtxt(args.base_path+"/"+args.robot_teacher+"/"+args.map_dataset+".csv", delimiter=',')
            target_new = np.genfromtxt(args.base_path+"/"+args.robot_teacher+"/"+args.map_dataset+"/manipulabilities.csv", delimiter=',')
            target_new = target_new.reshape((target_new.shape[0], 3, 3)) 
            # filename_manip_groundtruth=args.base_path+"/"+args.robot_student+"/"+args.map_dataset+".csv" 
            filename_manip_groundtruth=args.base_path+"/"+args.robot_student+"/"+args.map_dataset+"/manipulabilities.csv" 
            manip_groundtruth = np.genfromtxt(filename_manip_groundtruth, delimiter=',')
            manip_groundtruth = manip_groundtruth.reshape((manip_groundtruth.shape[0], 3, 3))   
            for i in np.arange(target_new.shape[0]):
                m=target_new[i]
                w,v = np.linalg.eigh(m)
                w[w<1e-12]=0.0001
                m=np.matmul(np.matmul(v, np.diag(w)), v.transpose())
                target_new[i]=m
            for i in np.arange(manip_groundtruth.shape[0]):
                m=manip_groundtruth[i]
                w,v = np.linalg.eigh(m)
                w[w<1e-12]=0.0001
                m=np.matmul(np.matmul(v, np.diag(w)), v.transpose())
                manip_groundtruth[i]=m
        else:
            target_new = np.genfromtxt(args.base_path+"/"+args.robot_teacher+"/"+args.map_dataset+"/manipulabilities_20.csv", delimiter=',')
            # target_new = np.genfromtxt(args.base_path+"/"+args.robot_teacher+"/"+args.map_dataset+"/manipulabilities_interpolated.csv", delimiter=',')
            filename_manip_groundtruth=args.base_path+"/"+args.robot_student+"/"+args.map_dataset+"/manipulabilities_20.csv" 
            # filename_manip_groundtruth=args.base_path+"/"+args.robot_student+"/"+args.map_dataset+"/manipulabilities_interpolated_groundtruth.csv" 
            target_new = target_new.reshape((target_new.shape[0], 3, 3))  
            # target_new = target_new[:,1:].reshape((target_new[:,1:].shape[0], 3, 3))  
            manip_groundtruth = np.genfromtxt(filename_manip_groundtruth, delimiter=',')
            manip_groundtruth = manip_groundtruth.reshape((manip_groundtruth.shape[0], 3, 3))   
            # manip_groundtruth = manip_groundtruth[:,1:].reshape((manip_groundtruth[:,1:].shape[0], 3, 3))   


        #####
        #manip_groundtruth = perform_transformation(target_new, T, R, s)
        #mg=np.zeros((manip_groundtruth.shape[0],10))
        #mg[:,1:]=np.reshape(manip_groundtruth, (manip_groundtruth.shape[0],9))

        #np.savetxt(args.base_path+"/"+args.robot_student+"/"+args.map_dataset+"/manipulabilities_interpolated_groundtruth.csv", mg, delimiter=",")
        ####

    R = np.genfromtxt(results_path+"/R_icp_"+args.robot_teacher+"_to_"+args.robot_student+".txt", delimiter=',')
    T = np.genfromtxt(results_path+"/T_icp_"+args.robot_teacher+"_to_"+args.robot_student+".txt", delimiter=',')
    s = np.genfromtxt(results_path+"/s_icp_"+args.robot_teacher+"_to_"+args.robot_student+".txt", delimiter=',')



    if R.ndim == 1:
        R = np.reshape(R, (1, 3, 3))
    else:
        R = np.reshape(R, (R.shape[0], 3, 3))
    T = np.reshape(T, (1, 3, 3))
    s=s.ravel()

    # for i in np.arange(target_new.shape[0]):
    #     print(np.linalg.eigvals(target_new[i]))

    fig2 = plt.figure(figsize=(16.5, 6))
    axs2 = list()
    axs2.append(fig2.add_subplot(1, 2, 1))
    axs2[0].set_title("\\textit{Input before transformation}")
    plot_diffusion_embedding_target_new(manip_groundtruth, target_new, axs2[0])


    print("ORIGINAL RMSE=%.3f"%(get_mean_dist_pairs(target_new, manip_groundtruth)))

    target_new = perform_transformation(source_org, target_new, T, R, s, args.map_dataset)

    #for m in target_new:
    #    print("eigs new: ", np.linalg.eigvalsh(m))


    if args.cv_k is None:
        if(args.robot_teacher=="rhuman" or args.robot_teacher=="2dof"):
            np.savetxt(args.base_path+"/"+args.robot_student+"/"+args.map_dataset+"/manipulabilities_mapped_icp.csv", np.reshape(target_new, (target_new.shape[0], 9)), delimiter=",")
            np.savetxt(final_results_path+"/validation/"+args.map_dataset+"/manipulabilities_mapped_icp.csv", np.reshape(target_new, (target_new.shape[0], 9)), delimiter=",")
            #target_mapped_naive = np.genfromtxt(args.base_path+"/"+args.robot_student+"/"+args.map_dataset+"/manipulabilities_mapped_naive.csv", delimiter=',', dtype=np.double) 
        else:
            np.savetxt(args.base_path+"/"+args.robot_student+"/"+args.map_dataset+"/manipulabilities_interpolated_mapped_icp.csv", np.reshape(target_new, (target_new.shape[0], 9)), delimiter=",")
            np.savetxt(final_results_path+"/validation/"+args.map_dataset+"/manipulabilities_interpolated_mapped_icp.csv", np.reshape(target_new, (target_new.shape[0], 9)), delimiter=",")
            #target_mapped_naive = np.genfromtxt(args.base_path+"/"+args.robot_student+"/"+args.map_dataset+"/manipulabilities_interpolated_mapped_naive.csv", delimiter=',', dtype=np.double)
    else:
        np.savetxt(args.base_path+"/"+args.robot_student+"/"+args.lookup_dataset+"/cv/manipulabilities_mapped_icp.csv", np.reshape(target_new, (target_new.shape[0], 9)), delimiter=",")
        #target_mapped_naive = np.genfromtxt(args.base_path+"/"+args.robot_student+"/"+args.lookup_dataset+"/cv/manipulabilities_mapped_naive.csv", delimiter=',', dtype=np.double)
    
    axs2.append(fig2.add_subplot(1, 2, 2))

    #plot naive mapping too in 2d embedding to be able to compare results
    #target_mapped_naive = target_mapped_naive.reshape((target_mapped_naive.shape[0], 3, 3))
    for i in target:
        assert(np.all(np.linalg.eigvals(i) > 0))
    for i in target_new:
        assert(np.all(np.linalg.eigvals(i) > 0))
    #for i in target_mapped_naive:
    #    assert(np.all(np.linalg.eigvals(i) > 0))

    axs2[1].set_title("Input after transformation")
    plot_diffusion_embedding_target_new(manip_groundtruth, target_new, axs2[1])

    fig2.tight_layout()
    fig2.savefig(results_path2+"/mapped.pdf")
    fig2.savefig(final_results_path+"/validation/"+args.map_dataset+"/mapped.pdf")

    #find_nearest_neighbors(manip_groundtruth, target_new) # just for error between source and target without mapping












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
