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
from plot_2d_embeddings import embed_2d, embed_2d_target_new, embed_2d_target_new_and_naive, plot_diffusion_embedding, \
    plot_diffusion_embedding_target_new, plot_diffusion_embedding_target_new_and_naive
import tkinter
import copy
from scipy.linalg import eigh
import matplotlib
import math
from path_sing2sing import find_singular_geodesic_paths, find_most_singular_points, find_most_singular_points_diff_dir

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
        dst_sum += ds
    mean_dist = float(dst_sum / target.shape[0])
    print("Mean dist between source and target pairs given: ", mean_dist)
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


def icp_iteration_sing2sing(source_org, target, dist, plotting=False):

    source_sing, target_sing, idx_s, idx_t = find_singular_geodesic_paths(source, target, 3) # num points to use

    print("USING %i SING2SING POINTS " %(idx_s.shape[0]))

    mean_target_sing = mean_riemann(target_sing)
    mean_source_sing = mean_riemann(source_sing)

    if plotting:
        fig3 = plt.figure(figsize=(20, 7))
        fig3.suptitle('Initial iteration process')
        axs3 = list()
        axs3.append(fig3.add_subplot(1, 5, 1))
        axs3[0].set_title("Original")
        plot_diffusion_embedding(source_org, target, axs3[0], idx_s, idx_t, pairwise=False)

    ### move target to id ###
    target = np.stack([np.dot(invsqrtm(mean_target_sing), np.dot(ti, invsqrtm(mean_target_sing))) for ti in target])  
    target_sing = np.stack([np.dot(invsqrtm(mean_target_sing), np.dot(ti, invsqrtm(mean_target_sing))) for ti in target_sing])  
    source_sing = np.stack([np.dot(invsqrtm(mean_source_sing), np.dot(si, invsqrtm(mean_source_sing))) for si in source_sing])  # move source to id

    if plotting:
        axs3.append(fig3.add_subplot(1, 5, 2))
        axs3[1].set_title("Recenter to id")
        plot_diffusion_embedding(source_org, target, axs3[1], idx_s, idx_t, pairwise=False)

    ### stretch target at id ###
    disp_source = np.sum([distance_riemann(covi, np.eye(3)) ** 2 for covi in source_sing]) / len(source_sing)  # get stretching factor
    disp_target = np.sum([distance_riemann(covi, np.eye(3)) ** 2 for covi in target_sing]) / len(target_sing)
    s = 1.0 / round(np.sqrt(disp_target / disp_source),4)
    target_sing = np.stack([powm(ti, s) for ti in target_sing])  # stretch target at id
    target = np.stack([powm(ti, s) for ti in target])  # stretch target at id

    if plotting:
        axs3.append(fig3.add_subplot(1, 5, 3))
        axs3[2].set_title("Stretch at id")
        plot_diffusion_embedding(source_org, target, axs3[2], idx_s, idx_t, pairwise=False)

    ### find R and rotate all targets ###
    itermax=100
    iter_curr=0
    mean_t=999
    mean_sing=999
    R=list()
    err_all=list()


    while (iter_curr < itermax) and (mean_sing > 1e-1*1):
        rotation_matrix_iter = get_rotation_matrix(M=source_sing, Mtilde=target_sing, weights=None, dist='rie', x=gen_orth(3))
        target_tmp = np.stack([np.dot(rotation_matrix_iter, np.dot(t, rotation_matrix_iter.T)) for t in target])

        mean_t=0
        mean_nns=0
        for i in np.arange(source.shape[0]):
            # mean_t+= distance_riemann(source[i], target[i])
            mean_nns+= distance_riemann(source_org[i], target_tmp[i])
        iter_curr+=1
        # mean_t =mean_t/target.shape[0]
        mean_sing =mean_sing/target_sing.shape[0]
        # mean_nns =mean_nns/target_nns.shape[0]
        print("Mean after iteration %i: %.3f" %(iter_curr, mean_sing))
        err_all.append(mean_sing)
        R.append(rotation_matrix_iter)

    if(iter_curr == itermax):
        print("[MAX ITER] Looking for smallest error because maxiter reached!")
        min_idx = err_all.index(min(err_all))
        print("best iteration found: %i" %(min_idx))
        rotation_matrix_iter=R[min_idx]
        
    else:
        rotation_matrix_iter = R[-1]

    target = np.stack([np.dot(rotation_matrix_iter, np.dot(t, rotation_matrix_iter.T)) for t in target])
    target_sing = np.stack([np.dot(rotation_matrix_iter, np.dot(t, rotation_matrix_iter.T)) for t in target_sing])

    if plotting:
        axs3.append(fig3.add_subplot(1, 5, 4))
        axs3[3].set_title("Rotate wrt subsamples")
        plot_diffusion_embedding(source_org, target, axs3[3], idx_s, idx_t, pairwise=False)
        #plot_diffusion_embedding(source, target, axs2[3])


    ### move to source ###
    target = np.stack([np.dot(sqrtm(mean_source_sing), np.dot(ti, sqrtm(mean_source_sing))) for ti in target]) 
    target_sing = np.stack([np.dot(sqrtm(mean_source_sing), np.dot(ti, sqrtm(mean_source_sing))) for ti in target_sing]) 

    if plotting:
        axs3.append(fig3.add_subplot(1, 5, 5))
        axs3[4].set_title("Move to source")
        plot_diffusion_embedding(source_org, target, axs3[4], idx_s, idx_t, pairwise=False)
        fig3.savefig(results_path2+"/icp_initial_iteration.pdf", bbox_inches='tight')

    return target, s, rotation_matrix_iter, [mean_target_sing, mean_source_sing]



def icp_iteration_most_singular(source_org, target, dist, plotting=False):

    #source_org, target, w, mean_dist_nns = find_nearest_neighbors(source_org, target, dist, filterdup=True)
    # source_sing, target_sing, idx_s, idx_t = find_most_singular_points_diff_dir(source, target, 10)
    source_sing, target_sing, idx_s, idx_t, w = find_most_singular_points(source, target, 12)
    print("USING %i MOST SINGULAR POINTS " %(idx_s.shape[0]))
    #source_nn = source_org[w == 1]
    #target_nn = target[w == 1]
    #idx_s=np.argwhere(w==1)
    #idx_t=np.argwhere(w==1)

    mean_target_sing = mean_riemann(target_sing)
    mean_source_sing = mean_riemann(source_sing)

    if plotting:
        fig3 = plt.figure(figsize=(20, 7))
        fig3.suptitle('Initial iteration process')
        axs3 = list()
        axs3.append(fig3.add_subplot(1, 5, 1))
        axs3[0].set_title("Original")
        plot_diffusion_embedding(source_org, target, axs3[0], idx_s, idx_t, pairwise=True)

    ### move target to id ###
    target = np.stack([np.dot(invsqrtm(mean_target_sing), np.dot(ti, invsqrtm(mean_target_sing))) for ti in target])  
    target_sing = np.stack([np.dot(invsqrtm(mean_target_sing), np.dot(ti, invsqrtm(mean_target_sing))) for ti in target_sing])  
    source_sing = np.stack([np.dot(invsqrtm(mean_source_sing), np.dot(si, invsqrtm(mean_source_sing))) for si in source_sing])  # move source to id

    if plotting:
        axs3.append(fig3.add_subplot(1, 5, 2))
        axs3[1].set_title("Recenter to id")
        plot_diffusion_embedding(source_org, target, axs3[1], idx_s, idx_t, pairwise=True)

    ### stretch target at id ###
    disp_source = np.sum([distance_riemann(covi, np.eye(3)) ** 2 for covi in source_sing]) / len(source_sing)  # get stretching factor
    disp_target = np.sum([distance_riemann(covi, np.eye(3)) ** 2 for covi in target_sing]) / len(target_sing)
    s = 1.0 / round(np.sqrt(disp_target / disp_source),4)
    target_sing = np.stack([powm(ti, s) for ti in target_sing])  # stretch target at id
    target = np.stack([powm(ti, s) for ti in target])  # stretch target at id

    if plotting:
        axs3.append(fig3.add_subplot(1, 5, 3))
        axs3[2].set_title("Stretch at id")
        plot_diffusion_embedding(source_org, target, axs3[2], idx_s, idx_t, pairwise=True)

    ### find R and rotate all targets ###
    itermax=100
    iter_curr=0
    mean_t=999
    mean_sing=999
    R=list()
    err_all=list()


    while (iter_curr < itermax) and (mean_sing > 1e-1*1):
        rotation_matrix_iter = get_rotation_matrix(M=source_sing, Mtilde=target_sing, weights=w, dist='rie', x=gen_orth(3))
        target_tmp = np.stack([np.dot(rotation_matrix_iter, np.dot(t, rotation_matrix_iter.T)) for t in target])

        mean_t=0
        mean_nns=0
        for i in np.arange(source.shape[0]):
            # mean_t+= distance_riemann(source[i], target[i])
            mean_nns+= distance_riemann(source_org[i], target_tmp[i])
        iter_curr+=1
        # mean_t =mean_t/target.shape[0]
        mean_sing =mean_sing/target_sing.shape[0]
        # mean_nns =mean_nns/target_nns.shape[0]
        print("Mean after iteration %i: %.3f" %(iter_curr, mean_sing))
        err_all.append(mean_sing)
        R.append(rotation_matrix_iter)

    if(iter_curr == itermax):
        print("[MAX ITER] Looking for smallest error because maxiter reached!")
        min_idx = err_all.index(min(err_all))
        print("best iteration found: %i" %(min_idx))
        rotation_matrix_iter=R[min_idx]
        
    else:
        rotation_matrix_iter = R[-1]

    target = np.stack([np.dot(rotation_matrix_iter, np.dot(t, rotation_matrix_iter.T)) for t in target])
    target_sing = np.stack([np.dot(rotation_matrix_iter, np.dot(t, rotation_matrix_iter.T)) for t in target_sing])

    if plotting:
        axs3.append(fig3.add_subplot(1, 5, 4))
        axs3[3].set_title("Rotate wrt subsamples")
        plot_diffusion_embedding(source_org, target, axs3[3], idx_s, idx_t, pairwise=True)
        #plot_diffusion_embedding(source, target, axs2[3])


    ### move to source ###
    target = np.stack([np.dot(sqrtm(mean_source_sing), np.dot(ti, sqrtm(mean_source_sing))) for ti in target]) 
    target_sing = np.stack([np.dot(sqrtm(mean_source_sing), np.dot(ti, sqrtm(mean_source_sing))) for ti in target_sing]) 

    if plotting:
        axs3.append(fig3.add_subplot(1, 5, 5))
        axs3[4].set_title("Move to source")
        plot_diffusion_embedding(source_org, target, axs3[4], idx_s, idx_t, pairwise=True)
        fig3.savefig(results_path2+"/icp_initial_iteration.pdf", bbox_inches='tight')

    return target, s, rotation_matrix_iter, [mean_target_sing, mean_source_sing]



def icp_iteration(source_org, target, dist, plotting=False):

    source_org_cp = copy.deepcopy(source_org)
    target_cp = copy.deepcopy(target)

    source_org, target, w, mean_dist_nns = find_nearest_neighbors(source_org, target, dist, filterdup=True)
    source_nn = source_org[w == 1]
    target_nn = target[w == 1]
    idx_s=np.argwhere(w==1)
    idx_t=np.argwhere(w==1)

    mean_target = mean_riemann(target_nn)
    mean_source_org = mean_riemann(source_nn)

    if plotting:
        fig3 = plt.figure(figsize=(20, 7))
        fig3.suptitle('Initial iteration process')
        axs3 = list()
        axs3.append(fig3.add_subplot(1, 5, 1))
        axs3[0].set_title("Original")
        plot_diffusion_embedding(source_org, target, axs3[0], idx_s, idx_t)

    ### move target to id ###
    target = np.stack([np.dot(invsqrtm(mean_target), np.dot(ti, invsqrtm(mean_target))) for ti in target])  
    target_cp = np.stack([np.dot(invsqrtm(mean_target), np.dot(ti, invsqrtm(mean_target))) for ti in target_cp])  
    source = np.stack([np.dot(invsqrtm(mean_source_org), np.dot(si, invsqrtm(mean_source_org))) for si in source_org])  # move source to id

    if plotting:
        axs3.append(fig3.add_subplot(1, 5, 2))
        axs3[1].set_title("Recenter to id")
        plot_diffusion_embedding(source, target, axs3[1])

    ### stretch target at id ###
    disp_source = np.sum([distance_riemann(covi, np.eye(3)) ** 2 for covi in source_nn]) / len(source_nn)  # get stretching factor
    disp_target = np.sum([distance_riemann(covi, np.eye(3)) ** 2 for covi in target_nn]) / len(target_nn)
    s = 1.0 / round(np.sqrt(disp_target / disp_source),4)
    target = np.stack([powm(ti, s) for ti in target])  # stretch target at id
    target_cp = np.stack([powm(ti, s) for ti in target_cp])  # stretch target at id

    if plotting:
        axs3.append(fig3.add_subplot(1, 5, 3))
        axs3[2].set_title("Stretch at id")
        plot_diffusion_embedding(source, target, axs3[2], idx_s, idx_t)

    ### find R and rotate all targets ###
    itermax=100
    iter_curr=0
    mean_t=999
    mean_nns=999
    R=list()
    err_all=list()


    while (iter_curr < itermax) and (mean_nns > 1e-1*1):
        rotation_matrix_iter = get_rotation_matrix(M=source_org, Mtilde=target, weights=w, dist='rie', x=gen_orth(3))
        target_tmp = np.stack([np.dot(rotation_matrix_iter, np.dot(t, rotation_matrix_iter.T)) for t in target])

        mean_t=0
        mean_nns=0
        for i in np.arange(source.shape[0]):
            # mean_t+= distance_riemann(source[i], target[i])
            mean_nns+= w[i]*distance_riemann(source_org[i], target_tmp[i])
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
    target_cp = np.stack([np.dot(rotation_matrix_iter, np.dot(t, rotation_matrix_iter.T)) for t in target_cp])

    if plotting:
        axs3.append(fig3.add_subplot(1, 5, 4))
        axs3[3].set_title("Rotate wrt subsamples")
        plot_diffusion_embedding(source, target, axs3[3], idx_s, idx_t)
        #plot_diffusion_embedding(source, target, axs2[3])


    ### move to source ###
    target = np.stack([np.dot(sqrtm(mean_source_org), np.dot(ti, sqrtm(mean_source_org))) for ti in target]) 
    target_cp = np.stack([np.dot(sqrtm(mean_source_org), np.dot(ti, sqrtm(mean_source_org))) for ti in target_cp]) 

    if plotting:
        axs3.append(fig3.add_subplot(1, 5, 5))
        axs3[4].set_title("Move to source")
        plot_diffusion_embedding(source_org, target, axs3[4], idx_s, idx_t)
        fig3.savefig(results_path2+"/icp_initial_iteration.pdf", bbox_inches='tight')

    return target_cp, s, rotation_matrix_iter, [mean_target, mean_source_org]


def perform_transformation(target, T, R, s):
    print(T.shape)
    print(R.shape)
    print(s.shape)

    for i in np.arange(R.shape[0]):

        # recenter to id
        target = np.stack([np.dot(invsqrtm(T[2*i]), np.dot(ti, invsqrtm(T[2*i]))) for ti in target])

        # stretch
        target = np.stack([powm(covi, s[i]) for covi in target])

        # rotate
        target = np.stack([np.dot(R[i], np.dot(t, R[i].T)) for t in target])

        # recenter to source (all Ti same)
        target = np.stack([np.dot(sqrtm(T[2*i+1]), np.dot(ti, sqrtm(T[2*i+1]))) for ti in target])

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

iter = 25
dist = "rie"

source = np.genfromtxt(args.base_path+"/"+args.robot_student+"/"+args.lookup_dataset+"/manipulabilities.csv", delimiter=',')
# source = np.genfromtxt(args.base_path+"/"+args.robot_student+"/"+args.lookup_dataset+"/manipulabilities_normalized.csv", delimiter=',')
target = np.genfromtxt(args.base_path+"/"+args.robot_teacher+"/"+args.lookup_dataset+"/manipulabilities.csv", delimiter=',')
# target = np.genfromtxt(args.base_path+"/"+args.robot_teacher+"/"+args.lookup_dataset+"/manipulabilities_normalized.csv", delimiter=',')

source_org = source.reshape((source.shape[0], 3, 3))
target_org = target.reshape((target.shape[0], 3, 3))


source = copy.deepcopy(source_org)
target = copy.deepcopy(target_org)

np.random.shuffle(target)

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
    T_list=list()
    s_list=list()

    print("RUNNING ICP 2")



    fig2 = plt.figure(figsize=(20, 7))
    fig2.suptitle('Diffusion Map')
    axs2 = list()
    axs2.append(fig2.add_subplot(1, iter + 2, 1))
    axs2[0].set_title("Original")
    plot_diffusion_embedding(source_org, target, axs2[0])

    
    print("After initial iter and NN search mean dist = 000 ")

    axs2.append(fig2.add_subplot(1, iter + 2, 2))
    axs2[1].set_title("Initial (000)" )
    # axs2[1].set_title("Initial (%.2f)" %(mean_dist))
    plot_diffusion_embedding(source_org, target, axs2[1])

    print("-------------------------------------------------------------")

    err_list=list()

    for iter_i in tqdm(np.arange(iter)):
        print("Iteration %i/%i" %(iter_i, iter))

        if iter_i == 0:
            # target, s_iter, rotation_matrix_iter, Ts = icp_iteration(source, target, dist, True)
            #target, s_iter, rotation_matrix_iter, Ts = icp_iteration_sing2sing(source, target, dist, True)
            target, s_iter, rotation_matrix_iter, Ts = icp_iteration_most_singular(source, target, dist, True)
        else:

            #target, s_iter, rotation_matrix_iter, Ts = icp_iteration_sing2sing(source, target, dist, False)
            target, s_iter, rotation_matrix_iter, Ts = icp_iteration_most_singular(source, target, dist, False)
            #target, s_iter, rotation_matrix_iter, Ts  = icp_iteration(source, target, dist, False)
            print("target back from icp iteration:", target.shape)


        R_list.append(rotation_matrix_iter)
        T_list.append(Ts[0])
        T_list.append(Ts[1])
        s_list.append(s_iter)


        axs2.append(fig2.add_subplot(1, iter + 2, iter_i + 3))
        plot_diffusion_embedding(source, target, axs2[iter_i + 2])

        source_tmp, target_tmp, _, _ = find_nearest_neighbors(source, target, dist, filterdup=False)
        err_iter=get_mean_dist_pairs(source_tmp, target_tmp)
        err_list.append(err_iter)
        axs2[iter_i + 2].set_title("%.2f" % (err_iter))

        if(err_iter <=1e-3*8):
            break

        # if(iter_i == iter-1):
        #     print("[MAX ITER OUTER LOOP] Looking for smallest error because maxiter reached!")
        #     min_idx = err_list.index(min(err_list))
        #     print("best iteration found: %i" %(min_idx))
        #     rotation_matrix_iter=R[min_idx]
        print("Current mdist between means: %.3f" % (distance_riemann(mean_riemann(source_org), mean_riemann(target))) )
        print("-------------------------------------------------------------")
    if iter == 0:
        R_list.append(np.eye(3))

    ######
    ### move target to id ###
    mean_source = mean_riemann(source)
    mean_target = mean_riemann(target)
    target = np.stack([np.dot(invsqrtm(mean_target), np.dot(ti, invsqrtm(mean_target))) for ti in target]) 
    T_list.append(mean_target) 


    ### stretch target at id ###
    disp_source = np.sum([distance_riemann(covi, np.eye(3)) ** 2 for covi in source]) / len(source)  # get stretching factor
    disp_target = np.sum([distance_riemann(covi, np.eye(3)) ** 2 for covi in target]) / len(target)
    s = 1.0 / round(np.sqrt(disp_target / disp_source),4)
    target = np.stack([powm(ti, s) for ti in target])  # stretch target at id
    s_list.append(s)

    # final moving to source mean
    target = np.stack([np.dot(sqrtm(mean_source), np.dot(ti, sqrtm(mean_source))) for ti in target]) 
    print("Current mdist between means: %.3f" % (distance_riemann(mean_riemann(source_org), mean_riemann(target))) )

    T_list.append(mean_source)

    ######



    R_arr=np.zeros((len(R_list),3,3))
    for i in np.arange(len(R_list)):
        R_arr[i]=R_list[i]

    T_arr=np.zeros((len(T_list),3,3))
    for i in np.arange(len(T_list)):
        T_arr[i]=T_list[i]

    s_arr=np.zeros((len(s_list),1))
    for i in np.arange(len(s_list)):
        s_arr[i]=s_list[i]

    source_tmp, target_tmp, _, _ = find_nearest_neighbors(source_org, target, dist, filterdup=False)
    print("AFTER ALL ITERATIONs MEAN DIST: %.3f" % (get_mean_dist_pairs(source_tmp, target_tmp)) )
    print(len(s_list))


    # save results
    if iter==0:
        #np.savetxt(results_path+"/R_icp_"+args.robot_teacher+"_to_"+args.robot_student+".txt", np.reshape(R, (1, 9)), delimiter=',')
        np.savetxt(results_path+"/R_icp_"+args.robot_teacher+"_to_"+args.robot_student+".txt", np.reshape(R_arr, (len(R_list), 9)), delimiter=',')
        np.savetxt(results_path+"/T_icp_"+args.robot_teacher+"_to_"+args.robot_student+".txt", np.reshape(T_arr, (1, 9)), delimiter=',')
        np.savetxt(results_path+"/s_icp_"+args.robot_teacher+"_to_"+args.robot_student+".txt", s_arr, delimiter=',')
    else:
        #np.savetxt(results_path+"/R_icp_"+args.robot_teacher+"_to_"+args.robot_student+".txt", np.reshape(R, (iter, 9)), delimiter=',')
        np.savetxt(results_path+"/R_icp_"+args.robot_teacher+"_to_"+args.robot_student+".txt", np.reshape(R_arr, (len(R_list), 9)), delimiter=',')
        np.savetxt(results_path+"/T_icp_"+args.robot_teacher+"_to_"+args.robot_student+".txt", np.reshape(T_arr, (len(T_list), 9)), delimiter=',')
        np.savetxt(results_path+"/s_icp_"+args.robot_teacher+"_to_"+args.robot_student+".txt", s_arr, delimiter=',')

    fig2.tight_layout()
    fig2.savefig(results_path2+"/icp_diffusion_map.pdf", bbox_inches='tight')

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

        target_new = np.genfromtxt(args.base_path+"/"+args.robot_teacher+"/"+args.map_dataset+"/manipulabilities_interpolated.csv", delimiter=',')
        target_new = target_new[:,1:].reshape((target_new[:,1:].shape[0], 3, 3))  
        filename_manip_groundtruth=args.base_path+"/"+args.robot_student+"/"+args.map_dataset+"/manipulabilities_interpolated_groundtruth.csv" 
        manip_groundtruth = np.genfromtxt(filename_manip_groundtruth, delimiter=',')
        manip_groundtruth = manip_groundtruth[:,1:].reshape((manip_groundtruth[:,1:].shape[0], 3, 3))   


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
    T = np.reshape(T, (T.shape[0], 3, 3))
    print("S", s.shape)
    s=s.ravel()


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
    plot_diffusion_embedding_target_new_and_naive(manip_groundtruth, target_new, target_mapped_naive, axs2[1])

    fig2.tight_layout()
    fig2.savefig(results_path2+"/mapped_diffusion_map.pdf")

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
