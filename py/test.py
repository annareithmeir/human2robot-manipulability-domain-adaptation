import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pandas as pd
from scipy.linalg import eigh
import math
from numpy import genfromtxt
from get_cov_ellipsoid import get_cov_ellipsoid
from pyriemann.utils.distance import distance_riemann, distance_wasserstein
import copy
from pyriemann.utils.base import invsqrtm, sqrtm, logm, expm, powm
import time
from path_sing2sing import find_singular_geodesic_paths


def distance_logeuc(A,B):
    diff = logm(A) - logm(B)
    return np.linalg.norm(diff, axis=(-2, -1), ord="fro", keepdims=False)


'''
ma1=np.array([           0.0839  , -0.0008 ,  -0.0347,
   -0.0008,    0.0080 ,   0.0034,
   -0.0347  ,  0.0034  ,  0.0214]).reshape(3,3) # after


md=np.array([    0.0159,         0 ,        0,
         0 ,   0.0159,        0,
         0   ,      0 ,   0.0159]).reshape(3,3)

mb=np.array([     0.0393,   -0.0371  , -0.0284,
   -0.0371  ,  0.0988,    0.0085,
   -0.0284  ,  0.0085  ,  0.1266]).reshape(3,3)


fig = plt.figure()
ax = plt.axes(projection='3d')

X2,Y2,Z2 = get_cov_ellipsoid(ma1, [0,0,0], 1)
ax.plot_wireframe(X2,Y2,Z2, color='green', alpha=0.05, label='first')

X2,Y2,Z2 = get_cov_ellipsoid(mb, [0,0,0], 1)
ax.plot_wireframe(X2,Y2,Z2, color='red', alpha=0.05, label='before')

X2,Y2,Z2 = get_cov_ellipsoid(md, [0,0,0], 1)
ax.plot_wireframe(X2,Y2,Z2, color='black', alpha=0.05, label='desired')

ax.set_zlim(-1,1)
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.legend()

plt.show()
'''


# # assumes make_pairs already called!
# def reject_max_distance(source, target, max_dist, dist='rie'):
#     idx_rm=list()
#     if dist=='rie':
#         for i in np.arange(source.shape[0]):
#             if distance_riemann(source[i], target[i]) > max_dist:
#                 idx_rm.append(i)
#     if dist=='was':
#         for i in np.arange(source.shape[0]):
#             if distance_wasserstein(source[i], target[i]) > max_dist:
#                 idx_rm.append(i)
#     source = np.delete(source, idx_rm, 0)
#     target = np.delete(target, idx_rm, 0)
#     return source, target


# # assumes make_pairs already called!
# def reject_median(source,target, dist='rie'):
#     distances=np.zeros(source.shape[0])
#     if dist=='rie':
#         for i in np.arange(source.shape[0]):
#             distances[i] = distance_riemann(source[i], target[i])
#     if dist=='was':
#         for i in np.arange(source.shape[0]):
#             distances[i] = distance_wasserstein(source[i], target[i])
#     idx = np.argsort(distances) # indices from small to large
#     idx = idx[:int(idx.shape[0]/2)] # only keep lower half
#     source = source[idx]
#     target = target[idx]
#     return source, target


# # rearrange target according to nns
# def make_pairs(source, target, nns):
#     target_rearranged = np.zeros(target.shape)
#     source_rearranged = np.zeros(target.shape)
#     pos=0
#     for i in np.arange(target.shape[0]):
#         if nns[i] != -1:
#             target_rearranged[pos] = target[nns[i]]
#             source_rearranged[pos] = source[i]
#             pos=pos+1

#     return source_rearranged[:pos], target_rearranged[:pos]


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
#     print("Found %i duplicates and %i unique target sample neighbors" %(np.argwhere(duplicates>1).shape[0], np.unique(nns).shape[0]))

#     for i in np.arange(duplicates.shape[0]):
#         if duplicates[i] > 1:
#             stmp=np.argwhere(nns==i).ravel()
#             d=np.zeros(stmp.shape[0])
#             for s_idx in np.arange(stmp.shape[0]):
#                 d[s_idx] = distance_riemann(source[stmp[s_idx]], target[i])

#             min_idx= np.argmin(d)
#             stmp = np.delete(stmp, min_idx)
#             nns[stmp] = -1

#     source, target = make_pairs(source, target, nns)

#     mean_dist_2 = 0.0
#     for (s,t) in zip(source, target):
#         mean_dist_2+= distance_riemann(s,t)
#     mean_dist_2 = mean_dist_2/target.shape[0]

#     print("Mean dist between nns found (after removing duplicates): ", mean_dist_2)

#     return source, target, mean_dist_2

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


def make_pairs(source, target, nns):
    target_rearranged = np.zeros((source.shape[0],3,3))

    for i in np.arange(target_rearranged.shape[0]):
        print(nns[i])
        print(target[nns[i]])
        target_rearranged[i] = target[nns[i]]

    return target_rearranged


def reject_median(source,target, w, dist='rie'):
    distances=np.zeros(source.shape[0])
    if dist=='rie':
        for i in np.arange(source.shape[0]):
            distances[i] = distance_riemann(source[i], target[i])
    if dist=='was':
        for i in np.arange(source.shape[0]):
            distances[i] = distance_wasserstein(source[i], target[i])
    idx = np.argsort(distances) # indices from small to large
    w[idx[int(idx.shape[0]/2):]] = 0 # only keep lower half
    return w


def find_singular_geodesic_paths_x(source, target):
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
    print("sing_idxs_min_s",sing_idxs_min_s)


    # find most singular samples in t
    sing_idxs=list()
    for i in np.arange(target.shape[0]):
        w, _ = np.linalg.eig(target[i])
        sing_idxs.append(min(w)/max(w)) # if ratio very small then singular

    sing_idxs_min_t = np.array(sing_idxs).argsort()[:num]
    print("sing_idxs_min_t", sing_idxs_min_t)


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
    print(corresp_idx_t)

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
    print(diff_matrix)
    print("diffmax ",diff_max_i)



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

        print("---")
        print(sing_idxs_min_s[n],sing_idxs_min_s[diff_max_i[n]])

        if {sing_idxs_min_s[n],sing_idxs_min_s[diff_max_i[n]]} in paths_done:
            print("PATH ALREADY DONE BEFORE!")
            continue
        a_s = source[sing_idxs_min_s[n]]
        b_s = source[sing_idxs_min_s[diff_max_i[n]]] # most different pointing to a_s in s
        #b_s = source[diff_max_i[n]]  # most different pointing to a_s in s
        a_t = target[corresp_idx_t[n]] # corresp to a_s
        b_t = target[corresp_idx_t[diff_max_i[n]]]
        print("a_s= %i, b_s= %i, a_t= %i, b_t= %i" %(sing_idxs_min_s[n],sing_idxs_min_s[diff_max_i[n]],corresp_idx_t[n],corresp_idx_t[diff_max_i[n]]))

        paths_done.append({sing_idxs_min_s[n],sing_idxs_min_s[diff_max_i[n]]})

        path_points_ti.append(a_t)
        path_points_si.append(a_s)

        path_idx_si.append(sing_idxs_min_s[n])
        path_idx_ti.append(corresp_idx_t[n])

        source_tmp = copy.deepcopy(source)
        target_tmp = copy.deepcopy(target)


        # logeuc as distance
        if sing_idxs_min_s[n] == sing_idxs_min_s[diff_max_i[n]]: # as same as bs
            print("a_s same as b_s")
            path_points_si.append(b_s)
            path_idx_si.append(sing_idxs_min_s[diff_max_i[n]])
        else:
            #eta=distance_logeuc(a_s, b_s)
            eta=distance_logeuc(a_s, b_s)/2

            print("ETA: ", eta)
            #construct path for s
            while True:
                d_a=list()
                d_b=list()
                d_a_filtered=list()
                etatmp=eta

                for i in np.arange(source_tmp.shape[0]):
                    d_a.append(distance_logeuc(a_s, source_tmp[i]))  
                print("d_a: ", d_a)
                time.sleep(3)
                while True:
                    d_a_filtered=list()

                    for d_i in np.arange(len(d_a)):
                        if (d_a[d_i] < etatmp) and (d_a[d_i] > 1e-6) and (d_i not in path_idx_si):
                            d_a_filtered.append(int(1))
                        else:
                            d_a_filtered.append(int(0))

                    print("dafiltered: ", d_a_filtered)
                    if sum(d_a_filtered)==0:
                        #etatmp+=eta
                        etatmp+=eta
                        print("new eta: ", etatmp)
                        time.sleep(3)
                    else:
                        break
                for d_a_filtered_i in np.arange(len(d_a_filtered)):
                    if d_a_filtered[d_a_filtered_i] == 1:
                        d_b.append(distance_logeuc(b_s, source_tmp[d_a_filtered_i]))
                    else:
                        d_b.append(9999)
                print("d_b",d_b)
                min_db=min(d_b)
                min_db_idx=d_b.index(min_db)
                print("min_db_idx ", min_db_idx)
                path_points_si.append(source_tmp[min_db_idx])
                path_idx_si.append(min_db_idx)
                if min_db < 1e-6: # b_t is reached
                    ("b_s is reached: ")
                    break
                else:
                    a_s = source_tmp[min_db_idx]
                    print("new a_s= %i" %(min_db_idx))
                    time.sleep(3)


        # ANGLE as distance
        # if sing_idxs_min_s[n] == sing_idxs_min_s[diff_max_i[n]]: # as same as bs
        #     print("a_s same as b_s")
        #     path_points_si.append(b_s)
        #     path_idx_si.append(sing_idxs_min_s[diff_max_i[n]])
        # else:
        #     #eta=distance_logeuc(a_s, b_s)
        #     #eta=distance_logeuc(a_s, b_s)/2
        #     #eta=0.992 #angle
        #     ### angle difference of singularity
        #     vt,wt = np.linalg.eig(a_s)
        #     w_min_as = wt[:,np.argmin(vt)] # eigvec corresp to smallest eigval

        #     vt,wt = np.linalg.eig(b_s)
        #     w_min_bs = wt[:,np.argmin(vt)] # eigvec corresp to smallest eigval
        #     eta = 1-((1-abs(w_min_as.dot(w_min_bs)))/3) #angle
        #     print("ETA: ", eta)
        #     #construct path for s
        #     while True:
        #         d_a=list()
        #         d_b=list()
        #         d_a_filtered=list()
        #         etatmp=eta

        #         ### angle difference of singularity
        #         vt,wt = np.linalg.eig(a_s)
        #         w_min_as = wt[:,np.argmin(vt)] # eigvec corresp to smallest eigval

        #         vt,wt = np.linalg.eig(b_s)
        #         w_min_bs = wt[:,np.argmin(vt)] # eigvec corresp to smallest eigval
                
        #         for i in np.arange(source_tmp.shape[0]):
        #             #d_a.append(distance_logeuc(a_s, source_tmp[i]))

        #             ### angle difference of singularity
        #             vt,wt = np.linalg.eig(source[i])
        #             w_min_s = wt[:,np.argmin(vt)] # eigvec corresp to smallest eigval
        #             d_a.append(abs(w_min_as.dot(w_min_s))) # collect all angle differences of most singular dimension
        #             ###
        #         print("d_a: ", d_a)
        #         time.sleep(3)
        #         while True:
        #             d_a_filtered=list()
        #             #da=[num for num in d_a if num > 1e-6 and num < etatmp and (d_a.index(num) not in path_idx_s) ]
        #             #for item in da:
        #             #    #d_a_filtered.append(d_a.index(item))
        #             #    d_a_filtered.append(np.where(np.array(d_a) == item)[0])
        #             #if len(d_a_filtered)==0:
        #             #    etatmp+=eta
        #             #else:
        #             #    break

        #             for d_i in np.arange(len(d_a)):
        #                 if (d_a[d_i] > etatmp) and (d_a[d_i] < 1-1e-6) and (d_i not in path_idx_si):
        #                     d_a_filtered.append(int(1))
        #                 else:
        #                     d_a_filtered.append(int(0))

        #             print("dafiltered: ", d_a_filtered)
        #             if sum(d_a_filtered)==0:
        #                 #etatmp+=eta
        #                 etatmp-=0.1
        #                 print("new eta: ", etatmp)
        #                 time.sleep(3)
        #             else:
        #                 break
        #         for d_a_filtered_i in np.arange(len(d_a_filtered)):
        #             if d_a_filtered[d_a_filtered_i] == 1:
        #                 #d_b.append(distance_logeuc(b_s, source_tmp[d_a_filtered_i]))
        #                 ### angle difference of singularity
        #                 vt,wt = np.linalg.eig(source_tmp[d_a_filtered_i])
        #                 w_min_s = wt[:,np.argmin(vt)] # eigvec corresp to smallest eigval
        #                 d_b.append(abs(w_min_s.dot(w_min_bs))) # collect all angle differences of most singular dimension
        #                 ###
        #             else:
        #                 #d_b.append(9999)
        #                 d_b.append(-9999) # angle
        #         print("d_b",d_b)
        #         #min_db=min(d_b)
        #         min_db=max(d_b) # for angle
        #         min_db_idx=d_b.index(min_db)
        #         print("min_db_idx ", min_db_idx)
        #         #print("will be added to path: source[]", d_a_filtered[min_db_idx])
        #         #path_points_si.append(source_tmp[d_a_filtered[min_db_idx]])
        #         #path_idx_si.append(d_a_filtered[min_db_idx])
        #         #if min_db < 1e-6: # b_t is reached
        #         #    ("b_s is reached: ")
        #         #    break
        #         #else:
        #         #    a_s = source_tmp[d_a_filtered[d_b.index(min_db)]]
        #         #    print("nes a_s= %i" %(d_a_filtered[d_b.index(min_db)]))
        #         #    #source_tmp = np.delete(source_tmp, d_a_filtered[d_b.index(min_db)],0)
        #         print("will be added to path: source[]", min_db_idx)
        #         path_points_si.append(source_tmp[min_db_idx])
        #         path_idx_si.append(min_db_idx)
        #         if min_db > 1-1e-6: # b_t is reached
        #             ("b_s is reached: ")
        #             break
        #         else:
        #             a_s = source_tmp[min_db_idx]
        #             print("new a_s= %i" %(min_db_idx))
        #             #source_tmp = np.delete(source_tmp, d_a_filtered[d_b.index(min_db)],0)
        #             time.sleep(3)


        if corresp_idx_t[n] == corresp_idx_t[diff_max_i[n]]:
            path_points_ti.append(b_t)
            path_idx_ti.append(corresp_idx_t[diff_max_i[n]])
        else:
            #eta=distance_logeuc(a_t, b_t)
            eta=distance_logeuc(a_t, b_t)/3
            #construct path for t
            while True:
                d_a=list()
                d_b=list()
                d_a_filtered=list()
                etatmp=eta
                
                for i in np.arange(target_tmp.shape[0]):
                    d_a.append(distance_logeuc(a_t, target_tmp[i]))
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
                    d_b.append(distance_logeuc(b_t, target_tmp[ii]))
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
        print(paths_done)

        print("NNS:\n", nns)
        print("path  idx s:\n", path_idx_s)
        print("path  idx t:\n", path_idx_t)


    print("TOTAL LENGHTS OF SING2SING PATHS")
    print(len(path_points_s))
    print(len(path_points_t))
    path_points_t=np.array(path_points_t)
    path_points_s=np.array(path_points_s)
    return path_points_s, path_points_t, np.array(path_idx_s), np.array(path_idx_t)


def find_nearest_neighbors(source, target, distance='rie'):
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
    print(nns)
    

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
            w[stmp] = 0

    print(nns)
    print(w)
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
    print(w)

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



import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D

#s = genfromtxt("/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/panda/10_new/manipulabilities_interpolated.csv", delimiter=',')
s = genfromtxt("/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/panda/reach_up/manipulabilities_interpolated.csv", delimiter=',')
s=s[1:,1:]
s=s.reshape((s.shape[0],3,3))

t = genfromtxt("/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/toy_data/reach_up/manipulabilities_interpolated_groundtruth.csv", delimiter=',')
#t = genfromtxt("/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/toy_data/10_new/manipulabilities_interpolated_groundtruth.csv", delimiter=',')
t=t[:,1:]
t=t.reshape((t.shape[0],3,3))

#s=np.repeat([np.eye(3)],10,0)
#t=np.repeat([np.eye(3)],10,0)
#for i in np.arange(10):
#	s[i]=gen_spd(3)
#	t[i]=gen_spd(3)


#s[0]=np.eye(3)*[10,4,0.1]
#s[1]=np.eye(3)*[0.1,4,10]
#s[2]=np.eye(3)*[8,0.1,5]

#t[0]=np.eye(3)*[5,0.1,8]
#t[1]=np.eye(3)*[0.1,4,10]
#t[2]=np.eye(3)*[10,4,0.1]





#reject_max_distance(s,t, 1.5)
#reject_median(s,t)
sr, tr, w, _ = find_nearest_neighbors(s,t)
#print("TR FOUND: \n", tr)
#print("SR FOUND: \n", sr)

#find_singular_geodesic_paths(s,t)
#path_points_s, path_points_t, path_idx_s, path_idx_t = find_singular_geodesic_paths(s,t,4)


fig = plt.figure()
ax = plt.axes(projection='3d')
plt.title('Mappings from Panda to Toy Data')
scaling_factor_plot=0.3
cnt=0

for i in np.arange(0,10,1):


    si=scaling_factor_plot*s[i]
    ti=scaling_factor_plot*t[i]

    X2,Y2,Z2 = get_cov_ellipsoid(si, [1*cnt,0,0], 1)
    ax.plot_wireframe(X2,Y2,Z2, color='blue', alpha=0.05)

    X2,Y2,Z2 = get_cov_ellipsoid(ti, [1*cnt,0,0], 1)
    ax.plot_wireframe(X2,Y2,Z2, color='red', alpha=0.05)

    cnt+=1

scale=np.diag([cnt, 1, 1, 1.0])
scale=scale*(1.0/scale.max())
scale[3,3]=0.7
def short_proj():
  return np.dot(Axes3D.get_proj(ax), scale)


blue_patch = mpatches.Patch(color='blue', label='s')
red_patch = mpatches.Patch(color='red', label='t')

plt.legend(handles=[ blue_patch, red_patch])

ax.get_proj=short_proj
ax.set_box_aspect(aspect = (1,1,1))

plt.xlim(-0.5, 10)
plt.ylim(-0.5, 0.5)
ax.set_zlim(-0.5, 0.5)

#plt.show()
