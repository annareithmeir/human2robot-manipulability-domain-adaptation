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

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import argparse

np.random.seed(0)
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
    #print(nns)
    

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
            nns[stmp] = -1

    source, target = make_pairs(source, target, nns)

    mean_dist_2 = 0.0
    for (s,t) in zip(source, target):
        if distance == 'rie':
            mean_dist_2+= distance_riemann(t, s)
        elif distance == 'fro':
            mean_dist_2+= LA.norm(logm(invsqrtm(t) * s * invsqrtm(t)), 'fro')
        elif distance == 'was':
            mean_dist_2+= distance_wasserstein(t,s)
        else:
            print("[ERROR] either rie, was or fro!")
            return nns
    mean_dist_2 = mean_dist_2/target.shape[0]


    #source, target = reject_median(source, target, distance)

    mean_dist_3 = 0.0
    #for (s,t) in zip(source, target):
    #    if distance == 'rie':
    #        mean_dist_3+= distance_riemann(t, s)
    ##    elif distance == 'fro':
    #        mean_dist_3+= LA.norm(logm(invsqrtm(t) * s * invsqrtm(t)), 'fro')
    #    elif distance == 'was':
    #        mean_dist_3+= distance_wasserstein(t,s)
    #    else:
    #        print("[ERROR] either rie, was or fro!")
    #        return nns
    #mean_dist_3 = mean_dist_3/target.shape[0]

    #print("After rejection %i pairs left." %(source.shape[0]))
    print("Mean dists between nns found (before/after removal, reject median): %.3f, %.3f, %.3f" %  (mean_dist, mean_dist_2, mean_dist_3))

    return source, target, mean_dist_3


def find_singular_geodesic_paths(source, target):
    s=list()
    t=list()

    path_points_t=list()
    path_points_s=list()
    path_idx_t=list()
    path_idx_s=list()

    num=2 # number of singular points to select
    print("SING2SING with %i points" %(num))

    # find most singular samples in s
    sing_idxs=list()
    for i in np.arange(source.shape[0]):
        w, _ = np.linalg.eig(source[i])
        #print(w)
        #print(min(w),max(w),max(w)/min(w))
        sing_idxs.append(max(w)/min(w)) # if ratio very small then singular

    sing_idxs_min_s = np.array(sing_idxs).argsort()[-num:]


    # find most singular samples in t
    sing_idxs=list()
    for i in np.arange(target.shape[0]):
        w, _ = np.linalg.eig(target[i])
        sing_idxs.append(min(w)/max(w)) # if ratio very small then singular

    sing_idxs_min_t = np.array(sing_idxs).argsort()[:num]


    # find most similar directing sample between s and t (dot product near 1)
    corresp_idx_t = np.zeros(num, dtype=int)
    for i in np.arange(num):
        si=sing_idxs_min_s[i]
        vs,ws = np.linalg.eig(source[si])
        w_min_s = ws[:,np.argmin(vs)] # eigvec corresp to smallest eigval
        angles=list()
        for j in sing_idxs_min_t:
            vt,wt = np.linalg.eig(target[j])
            w_min_t = wt[:,np.argmin(vt)] # eigvec corresp to smallest eigval
            #angle = arccos(dot(A,B) / (|A|* |B|))
            angles.append(w_min_s.dot(w_min_t)) # collect all angles
            #angles.append(math.acos(w_min_s.dot(w_min_t)/(np.linalg.norm(w_min_s,ord=1)*np.linalg.norm(w_min_t,ord=1)))) # collect all angles
        angle_min = np.argmax(abs(np.array(angles))) #select smallest angle
        t_max = target[sing_idxs_min_t[angle_min]] # source[i] and t_max are pair of similar pointing sing matrices
        corresp_idx_t[i] = sing_idxs_min_t[angle_min]

    # find most different pointing samples within s
    diff_matrix=np.ones((num,num))*999
    for i in np.arange(num):
        si=sing_idxs_min_s[i]
        vi,wi = np.linalg.eig(source[si])
        w_min_i = wi[:,np.argmin(vi)] # eigvec corresp to smallest eigval

        for j in np.arange(num):
            if (i != j):
                sj=sing_idxs_min_s[j]
                #print(sj)
                vj,wj = np.linalg.eig(source[sj])
                #print(vj)
                #print(wj)
                w_min_j = wj[:,np.argmin(vj)] # eigvec corresp to smallest eigval
                #print(np.argmin(vj))
                #print(w_min_j)
                diff_matrix[i,j] = w_min_i.dot(w_min_j)

    diff_max_i = np.argmin(abs(diff_matrix), axis=1) # create path between each of them



    # create paths for each pair (if not already in othetr direction done)
    paths_done=list()

    path_points_si=list()
    path_points_ti=list()
    path_idx_si=list()
    path_idx_ti=list()
    for n in np.arange(num):
        path_points_si.clear()
        path_points_ti.clear()
        path_idx_si.clear()
        path_idx_ti.clear()

        if {sing_idxs_min_s[n],sing_idxs_min_s[diff_max_i[n]]} in paths_done:
            continue
        a_s = source[sing_idxs_min_s[n]]
        b_s = source[diff_max_i[n]] # most different pointing to a_s in s
        a_t = target[corresp_idx_t[n]] # corresp to a_s
        b_t = target[corresp_idx_t[diff_max_i[n]]]

        paths_done.append({sing_idxs_min_s[n],diff_max_i[n]})

        path_points_ti.append(a_t)
        path_points_si.append(a_s)

        path_idx_si.append(sing_idxs_min_s[n])
        path_idx_ti.append(corresp_idx_t[n])

        source_tmp = copy.deepcopy(source)
        target_tmp = copy.deepcopy(target)

        if sing_idxs_min_s[n] == diff_max_i[n]: # as same as bs
            path_points_si.append(b_s)
            path_idx_si.append(diff_max_i[n])
        else:
            eta=distance_riemann(a_s, b_s)
            #eta=distance_logeuc(a_s, b_s)/2
            #eta=distance_logeuc(a_s, b_s)/10
            #construct path for s
            while True:
                d_a=list()
                d_b=list()
                d_a_filtered=list()
                etatmp=eta
                
                for i in np.arange(source_tmp.shape[0]):
                    d_a.append(distance_riemann(a_s, source_tmp[i]))
                    #d_a.append(distance_logeuc(a_s, source_tmp[i]))
                while True:
                    da=[num for num in d_a if num > 1e-6 and num < etatmp]
                    for item in da:
                        d_a_filtered.append(d_a.index(item))
                    if len(d_a_filtered)==0:
                        etatmp+=eta
                    else:
                        break
                for ii in d_a_filtered:
                    d_b.append(distance_riemann(b_s, source_tmp[ii]))
                    #d_b.append(distance_logeuc(b_s, source_tmp[ii]))
                min_db=min(d_b)
                path_points_si.append(source_tmp[d_a_filtered[d_b.index(min_db)]])
                path_idx_si.append(d_a_filtered[d_b.index(min_db)])
                if min_db < 1e-6: # b_t is reached
                    break
                else:
                    a_s = source_tmp[d_a_filtered[d_b.index(min_db)]]
                    source_tmp = np.delete(source_tmp, d_a_filtered[d_b.index(min_db)],0)


        if corresp_idx_t[n] == corresp_idx_t[diff_max_i[n]]:
            path_points_ti.append(b_t)
            path_idx_ti.append(corresp_idx_t[diff_max_i[n]])
        else:
            eta=distance_riemann(a_t, b_t)
            #eta=distance_logeuc(a_t, b_t)/2
            #eta=distance_logeuc(a_t, b_t)/10
            #construct path for t
            while True:
                d_a=list()
                d_b=list()
                d_a_filtered=list()
                etatmp=eta
                
                for i in np.arange(target_tmp.shape[0]):
                    #d_a.append(distance_logeuc(a_t, target_tmp[i]))
                    d_a.append(distance_riemann(a_t, target_tmp[i]))
                while True:
                    da=[num for num in d_a if num > 1e-6 and num < etatmp]
                    for item in da:
                        d_a_filtered.append(d_a.index(item))
                    if len(d_a_filtered)==0:
                        etatmp+=eta
                    else:
                        break
                etatmp=eta
                for ii in d_a_filtered:
                    d_b.append(distance_riemann(b_t, target_tmp[ii]))
                    #d_b.append(distance_logeuc(b_t, target_tmp[ii]))
                min_db=min(d_b)
                path_points_ti.append(target_tmp[d_a_filtered[d_b.index(min_db)]])
                path_idx_ti.append(d_a_filtered[d_b.index(min_db)])
                if min_db < 1e-6: # b_t is reached
                    break
                else:
                    a_t = target_tmp[d_a_filtered[d_b.index(min_db)]]
                    target_tmp = np.delete(target_tmp, d_a_filtered[d_b.index(min_db)],0)


        source_tmp = np.array(path_points_si)
        target_tmp = np.array(path_points_ti)

        nns = np.zeros(source_tmp.shape[0], dtype=np.int)  # for each target search smallest source

        dst_sum = 0
        for i in np.arange(source_tmp.shape[0]):
            if i== 0: # map a_s to a_t
                nns[i]=0
            if i== source_tmp.shape[0]-1: # map a_t to b_t
                nns[i] = target_tmp.shape[0]-1
            else:
                s=source_tmp[i]
                dist_min = 9999
                idx_s = 0
                for t in target_tmp:
                    ds = distance_riemann(t, s)
                    if ds < dist_min:
                        dist_min = ds
                        nns[i] = int(idx_s)
                    idx_s += 1
                dst_sum += dist_min
        mean_dist = float(dst_sum / target_tmp.shape[0])

        target_rearranged = np.zeros((source_tmp.shape[0],3,3))

        for i in np.arange(target_rearranged.shape[0]):
            target_rearranged[i] = target_tmp[nns[i]]


        for t in target_rearranged:
            path_points_t.append(t)

        for s in source_tmp:
            path_points_s.append(s)

        for t in path_idx_ti:
            path_idx_t.append(t)

        for s in path_idx_si:
            path_idx_s.append(s)

        print("path_t: %i, path_s: %i, idx_s: %i, idx_t: %i" %(len(path_points_t), len(path_points_s), len(path_idx_s), len(path_idx_t)))
        print(path_idx_s)
        print(path_idx_t)


    print("TOTAL LENGHTS OF SING2SING PATHS")
    print(len(path_points_s))
    print(len(path_points_t))
    path_points_t=np.array(path_points_t)
    path_points_s=np.array(path_points_s)
    return path_points_s, path_points_t, np.array(path_idx_s), np.array(path_idx_t)





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
    source_s2s, target_s2s, idx_s, idx_t = find_singular_geodesic_paths(source, target)
    print("SING2SING: ",source_s2s.shape[0])
    print("idx_s ", idx_s.shape[0])
    print(idx_s)
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

        R = get_rotation_matrix(M=source_s2s, Mtilde=target_s2s, weights=None, dist='rie', x=gen_orth(3))
        #R = get_rotation_matrix(M=source_subsample_id, Mtilde=target_subsample_id, weights=None, dist='rie', x=gen_orth(3))
        #R = get_rotation_matrix(M=source, Mtilde=target, weights=None, dist='rie', x=gen_orth(3))
        #R = get_rotation_matrix(M=source_nns, Mtilde=target_nns, weights=None, dist='rie', x=gen_orth(3))
        #print(R[1])
        #R=R[0] # tuple due to verboselevel=2 in solve()
        #print("\nRotation found: \n", R)

        # apply rotation to all samples
        target_s2s = np.stack([np.dot(R, np.dot(t, R.T)) for t in target_s2s]) 
        #target_nns = np.stack([np.dot(R, np.dot(t, R.T)) for t in target_nns]) 
        #target_subsample_id = np.stack([np.dot(R, np.dot(t, R.T)) for t in target_subsample_id]) 
        target = np.stack([np.dot(R, np.dot(t, R.T)) for t in target]) 
        R_all.append(R)

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



def recenter_to_id_stretch_rotate_recenter_to_source(source_org, target, dist):
    mean_target = mean_riemann(target)
    mean_source_org = mean_riemann(source_org)

    ### move target to id ###
    target = np.stack([np.dot(invsqrtm(mean_target), np.dot(ti, invsqrtm(mean_target))) for ti in target])  
    source = np.stack([np.dot(invsqrtm(mean_source_org), np.dot(si, invsqrtm(mean_source_org))) for si in source_org])  # move source to id

    ### stretch target at id ###
    disp_source = np.sum([distance_riemann(covi, np.eye(3)) ** 2 for covi in source]) / len(source)  # get stretching factor
    disp_target = np.sum([distance_riemann(covi, np.eye(3)) ** 2 for covi in target]) / len(target)
    s = 1.0 / round(np.sqrt(disp_target / disp_source),4)
    target = np.stack([powm(ti, s) for ti in target])  # stretch target at id

    ### move target to id (slightly changed due to stretching) ###
    target = np.stack([np.dot(invsqrtm(mean_riemann(target)), np.dot(ti, invsqrtm(mean_riemann(target)))) for ti in target])  # move target to id

    # subsampling 
    #idx_s, idx_t = find_subsample_idx(source, target)
    #target_subsample_id = target[idx_t]
    #source_subsample_id = source[idx_s]
    #source_subsample_id, target_subsample_id, _ = find_nearest_neighbors(source_subsample_id, target_subsample_id, dist)
    

    ### find R and rotate all targets ###
    itermax=100
    iter_curr=0
    mean_t=999
    #R_all=np.ones((3,3))
    R_all=list()
    while (iter_curr < itermax) and (mean_t > 1e-1*8):
        source_nns, target_nns, mean_dist_nns = find_nearest_neighbors(source, target, dist)
        #rotation_matrix_iter = get_rotation_matrix(M=source_subsample_id, Mtilde=target_subsample_id, weights=None, dist='rie', x=gen_orth(3))
        rotation_matrix_iter = get_rotation_matrix(M=source_nns, Mtilde=target_nns, weights=None, dist='rie', x=gen_orth(3))
        
        target = np.stack([np.dot(rotation_matrix_iter, np.dot(t, rotation_matrix_iter.T)) for t in target])
        target_nns = np.stack([np.dot(rotation_matrix_iter, np.dot(t, rotation_matrix_iter.T)) for t in target_nns])
        R_all.append(rotation_matrix_iter)

        mean_t=0
        for i in np.arange(source.shape[0]):
            #print(distance_riemann(source[i], target[i]))
            mean_t+= distance_riemann(source[i], target[i])
        #####
        iter_curr+=1
        mean_t =mean_t/target.shape[0]
        print("Mean after iteration %i: %.3f" %(iter_curr, mean_t))
    print("Mean Distance reached after iteration %i, %.3f" %(iter_curr, mean_t))

    ### move to source ###
    target = np.stack([np.dot(sqrtm(mean_source_org), np.dot(ti, sqrtm(mean_source_org))) for ti in target]) 

    return target, s, R_all


def perform_transformation(target, T, R, s):
    #assert(len(T)==len(R)==len(s))

    # recenter to id
    mean_target = mean_riemann(target)
    target = np.stack([np.dot(invsqrtm(mean_target), np.dot(ti, invsqrtm(mean_target))) for ti in target])

    # stretch
    target = np.stack([powm(covi, s[0]) for covi in target])

    for i in np.arange(len(R)):
        ### move target to id (slightly changed due to stretching) ###
        #target = np.stack([np.dot(invsqrtm(mean_riemann(target)), np.dot(ti, invsqrtm(mean_riemann(target)))) for ti in target])  # move target to id
        # rotate
        target = np.stack([np.dot(R[i], np.dot(t, R[i].T)) for t in target])

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

iter = 0
dist = "rie"

source = np.genfromtxt(args.base_path+"/"+args.robot_student+"/"+args.lookup_dataset+"/manipulabilities.csv", delimiter=',')
target = np.genfromtxt(args.base_path+"/"+args.robot_teacher+"/"+args.lookup_dataset+"/manipulabilities.csv", delimiter=',')

source_org = source.reshape((source.shape[0], 3, 3))
target_org = target.reshape((target.shape[0], 3, 3))

#####
# source_org=source_org
# target_org=target_org
# mean_source_org = mean_riemann(source_org)

# #for i in np.arange(3):
# #    source_org[i+3]=np.dot(sqrtm(mean_source_org), np.dot(source_org[i], sqrtm(mean_source_org)))



# R = [np.array([[-0.35047963,  0.87345198, -0.33800246],
#  [-0.7128812,  -0.48285521, -0.50857767],
#  [ 0.60742442, -0.06270949, -0.79189841]])]

# T = [np.array([[0.27200416, 0.06000769, 0.17850028],
#  [0.06000769, 0.53767129, 0.13022352],
#  [0.17850028, 0.13022352, 0.8615307 ]])]

# #R=[np.eye(3)]
# print("Using R\n:")
# print(R)
# print(np.dot(R[0],R[0].T))

# #T=[np.zeros((3,3))]
# #T[0][0][0] =0.1
# #T[0][1][1] =0.1
# #T[0][2][2] =0.1


# source_org=source_org
# #T[0]=mean_riemann(source_org)
# print("Using T\n:")
# print(T)

# s = [0.813] 
# #s=[1]

# target_org = copy.deepcopy(source_org)
# target_org = perform_transformation(target_org, T, R, s)

# # one point only
# #target_org=target_org[-2:]

# print("--")
# print("test dataset:")
# print(source_org)
# print(target_org)
# print("--")
# print("Means (should be same):")
# print(mean_riemann(source_org), "\n",mean_riemann(target_org))
# print("--")
# print("Distances:")
# for i in np.arange(source_org.shape[0]):
#     #print(i)
#     print(distance_riemann(source_org[i], target_org[i]))
#     #print("--")
#     #for j in np.arange(6):
#     #    print(distance_riemann(source_org[i], target_org[j]))

# print("Testing points (trafo from target to source back")
# print(source_org)
# print(perform_transformation(target_org, [mean_riemann(source_org)], [R[0].T], s))


# run_map_new=True
# run_map_find=True

#####


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
        R,_,_,target = initial_iteration(source_org, target, results_path2, dist)
    

    for r in R:
        R_list.append(r)

    ### find nns ###
    #source_nns, target_nns, _ = find_nearest_neighbors(source_org, target, dist)

    axs2.append(fig2.add_subplot(1, iter + 2, 2))
    axs2[1].set_title("Initial (%.2f)" %(get_mean_dist_pairs(source_org, target)))
    plot_diffusion_embedding(source_org, target, axs2[1])

    print("-------------------------------------------------------------")

    for iter_i in tqdm(np.arange(iter)):
        print("Iteration %i/%i" %(iter_i, iter))

        target, s_iter, rotation_matrix_iter = recenter_to_id_stretch_rotate_recenter_to_source(source_org, target, dist)

        assert(target.shape[0]==target_org.shape[0])
        for r in rotation_matrix_iter:
            R_list.append(r)
    
        #R[iter_i] = rotation_matrix_iter
        s[iter_i] = s_iter
        T[iter_i] = mean_riemann(source_org)

        #print("len(R): ", len(R_list))
        #print("s: ", s[iter_i])
        #print("T: ", T[iter_i])

        axs2.append(fig2.add_subplot(1, iter + 2, iter_i + 3))
        plot_diffusion_embedding(source_org, target, axs2[iter_i + 2])
        axs2[iter_i + 2].set_title("%.2f" % (get_mean_dist_pairs(source_org, target)))

        print("-------------------------------------------------------------")

    #####

    #print("---ALL DONE---")
    #for i in np.arange(source.shape[0]):
    #    print(distance_riemann(source_org[i], target[i]))
    #####

    R_arr=np.zeros((len(R_list),3,3))
    for i in np.arange(len(R_list)):
        R_arr[i]=R_list[i]
        #print(R_list[i])


    # save results
    if iter==0:
        #np.savetxt(results_path+"/R_icp_"+args.robot_teacher+"_to_"+args.robot_student+".txt", np.reshape(R, (1, 9)), delimiter=',')
        np.savetxt(results_path+"/R_icp_"+args.robot_teacher+"_to_"+args.robot_student+".txt", np.reshape(R_arr, (len(R_list), 9)), delimiter=',')
        np.savetxt(results_path+"/T_icp_"+args.robot_teacher+"_to_"+args.robot_student+".txt", np.reshape(T, (1, 9)), delimiter=',')
        np.savetxt(results_path+"/s_icp_"+args.robot_teacher+"_to_"+args.robot_student+".txt", s, delimiter=',')
    else:
        #np.savetxt(results_path+"/R_icp_"+args.robot_teacher+"_to_"+args.robot_student+".txt", np.reshape(R, (iter, 9)), delimiter=',')
        np.savetxt(results_path+"/R_icp_"+args.robot_teacher+"_to_"+args.robot_student+".txt", np.reshape(R_arr, (len(R_list), 9)), delimiter=',')
        np.savetxt(results_path+"/T_icp_"+args.robot_teacher+"_to_"+args.robot_student+".txt", np.reshape(T, (iter, 9)), delimiter=',')
        np.savetxt(results_path+"/s_icp_"+args.robot_teacher+"_to_"+args.robot_student+".txt", s, delimiter=',')

    fig2.tight_layout()
    fig2.savefig(results_path2+"/icp_diffusion_map.pdf", bbox_inches='tight')

print("-------------------------------------------------------------")
print("-------------------------------------------------------------")

# apply trafo to new points
if run_map_new:
    print("------> RUNNING NEW DATA")
    if args.cv_k is None:
        target_new = np.genfromtxt(args.base_path+"/"+args.robot_teacher+"/"+args.map_dataset+"/manipulabilities_interpolated.csv", delimiter=',')
        target_new = target_new[1:,1:].reshape((target_new[1:,1:].shape[0], 3, 3))   
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


    if iter<2:
        R = np.reshape(R, (R.shape[0], 3, 3))
        T = np.reshape(T, (1, 3, 3))
    else:
        print(R.shape)
        R = np.reshape(R, (R.shape[0], 3, 3))
        T = np.reshape(T, (T.shape[0], 3, 3))
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
