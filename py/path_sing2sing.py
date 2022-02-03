import numpy as np
from scipy.linalg import eigh
import math
from get_cov_ellipsoid import get_cov_ellipsoid
from pyriemann.utils.distance import distance_riemann, distance_wasserstein
import copy
from pyriemann.utils.base import invsqrtm, sqrtm, logm, expm, powm
import time


def distance_logeuc(A,B):
    diff = logm(A) - logm(B)
    return np.linalg.norm(diff, axis=(-2, -1), ord="fro", keepdims=False)


def find_singular_geodesic_path_angle(source, n, start_idx):
    path_points_si=list()
    path_idx_si=list()

    path_points_si.append(a_s)
    path_idx_si.append(start_idx)
    source_tmp = copy.deepcopy(source)



    # ANGLE as distance
    if sing_idxs_min_s[n] == sing_idxs_min_s[diff_max_i[n]]: # as same as bs
        print("a_s same as b_s")
        path_points_si.append(b_s)
        path_idx_si.append(sing_idxs_min_s[diff_max_i[n]])
    else:
        vt,wt = np.linalg.eig(a_s)
        w_min_as = wt[:,np.argmin(vt)] # eigvec corresp to smallest eigval

        vt,wt = np.linalg.eig(b_s)
        w_min_bs = wt[:,np.argmin(vt)] # eigvec corresp to smallest eigval
        eta = 1-((1-abs(w_min_as.dot(w_min_bs)))/10) #angle
        print("ETA: ", eta)
        #construct path for s
        while True:
            d_a=list()
            d_s_b=list()
            d_b=list()
            d_a_filtered=list()
            etatmp=eta

            ### angle difference of singularity
            vt,wt = np.linalg.eig(a_s)
            w_min_as = wt[:,np.argmin(vt)] # eigvec corresp to smallest eigval

            vt,wt = np.linalg.eig(b_s)
            w_min_bs = wt[:,np.argmin(vt)] # eigvec corresp to smallest eigval
            
            for i in np.arange(source_tmp.shape[0]):
                vt,wt = np.linalg.eig(source[i])
                w_min_s = wt[:,np.argmin(vt)] # eigvec corresp to smallest eigval
                d_a.append(abs(w_min_as.dot(w_min_s))) # collect all angle differences from a to si
                d_s_b.append(abs(w_min_s.dot(w_min_bs))) # collect all angle differences from si to b

            #print("d_a: ", d_a)
            dist_a_b = abs(w_min_as.dot(w_min_bs))
            while True:
                d_a_filtered=list()

                for d_i in np.arange(len(d_a)):
                    if (d_a[d_i] > etatmp) and (d_a[d_i] < 1-1e-6) and (d_i not in path_idx_si) and (dist_a_b <= d_s_b[d_i]):
                        d_a_filtered.append(int(1))
                    else:
                        d_a_filtered.append(int(0))

                #print("dafiltered: ", d_a_filtered)
                if sum(d_a_filtered)==0:
                    etatmp-=0.1
                    print("new eta: ", etatmp)
                else:
                    break
            for d_a_filtered_i in np.arange(len(d_a_filtered)):
                if d_a_filtered[d_a_filtered_i] == 1:
                    vt,wt = np.linalg.eig(source_tmp[d_a_filtered_i])
                    w_min_s = wt[:,np.argmin(vt)] # eigvec corresp to smallest eigval
                    d_b.append(abs(w_min_s.dot(w_min_bs))) # collect all angle differences of most singular dimension
                else:
                    d_b.append(-9999) # angle
            #print("d_b",d_b)
            min_db=max(d_b) # for angle
            min_db_idx=d_b.index(min_db)
            #print("min_db_idx ", min_db_idx)
            path_points_si.append(source_tmp[min_db_idx])
            path_idx_si.append(min_db_idx)
            if min_db > 1-1e-6: # b_t is reached
                ("b_s is reached: ")
                break
            else:
                a_s = source_tmp[min_db_idx]
                #print("new a_s= %i" %(min_db_idx))
    return path_points_si, path_idx_si


def find_singular_geodesic_path_logeuc(source, a_s, b_s, start_idx):
    path_points_si = list()
    path_idx_si = list()

    path_points_si.append(a_s)

    path_idx_si.append(start_idx)
    source_tmp = copy.deepcopy(source)


    eta=distance_logeuc(a_s, b_s)/3 # find b right away

    print("ETA set to: ", eta)
    #construct path for s
    while True:
        d_a=list()
        d_b=list()
        d_a_filtered=list()
        etatmp=eta

        dist_ab = distance_logeuc(a_s, b_s)

        for i in np.arange(source_tmp.shape[0]):
            d_a.append(distance_logeuc(a_s, source_tmp[i]))  
            #print(distance_logeuc(source_tmp[i], b_s))
        #print("d_a: ", d_a)
        while True:
            d_a_filtered=list()

            for d_i in np.arange(len(d_a)):
                if (d_a[d_i] < etatmp) and (d_a[d_i] > 1e-6) and (d_i not in path_idx_si) and (distance_logeuc(source_tmp[d_i], b_s)<=dist_ab):
                    d_a_filtered.append(int(1))
                else:
                    d_a_filtered.append(int(0))

            #print("dafiltered: ", d_a_filtered)
            if sum(d_a_filtered)==0:
                etatmp+=eta
                print("new eta: ", etatmp)
            else:
                break
        for d_a_filtered_i in np.arange(len(d_a_filtered)):
            if d_a_filtered[d_a_filtered_i] == 1:
                d_b.append(distance_logeuc(b_s, source_tmp[d_a_filtered_i]))
            else:
                d_b.append(9999)
        #print("d_b",d_b)
        min_db=min(d_b)
        min_db_idx=d_b.index(min_db)
        #print("min_db_idx ", min_db_idx)
        path_points_si.append(source_tmp[min_db_idx])
        path_idx_si.append(min_db_idx)
        #print("Current path idx: ", path_idx_si)
        if min_db < 1e-6: # b_t is reached
            ("b_s is reached: ")
            break
        else:
            a_s = source_tmp[min_db_idx]
            #print("new a_s= %i" %(min_db_idx))
    return path_points_si, path_idx_si


def find_most_singular_points(source, target, num, with_iso=True):
    s=list()
    t=list()

    path_points_t=list()
    path_points_s=list()
    path_idx_t=list()
    path_idx_s=list()

    print("SING2SING with %i points" %(num))

    # find most singular samples in s
    sing_idxs=list()
    for i in np.arange(source.shape[0]):
        w, _ = np.linalg.eig(source[i])
        sing_idxs.append(max(w)/min(w)) # if ratio very big then singular

    sing_idxs_min_s = np.array(sing_idxs).argsort()[-num:]
    iso_idx_s = np.array(sing_idxs).argsort()[0] # most isotropic sample
    print("most isotropic samples found: ",np.sort(np.array(sing_idxs))[0] )


    # find most singular samples in t
    sing_idxs=list()
    for i in np.arange(target.shape[0]):
        w, _ = np.linalg.eig(target[i])
        sing_idxs.append(max(w)/min(w)) # if ratio very small then singular

    sing_idxs_min_t = np.array(sing_idxs).argsort()[-num:]
    iso_idx_t = np.array(sing_idxs).argsort()[0]
    print("most isotropic samples found: ",np.sort(np.array(sing_idxs))[0] )

    v1,w1 = np.linalg.eig(source[iso_idx_s])
    w_min_1 = w1[:,np.argmin(v1)] # eigvec corresp to smallest eigval
    v2,w2 = np.linalg.eig(target[iso_idx_t])
    w_min_2 = w2[:,np.argmin(v2)] # eigvec corresp to smallest eigval
    print("Angle between isotropic samples: %.3f " %(w_min_1.dot(w_min_2)))
    iso_angle = abs(w_min_1.dot(w_min_2))


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
            angles.append(w_min_s.dot(w_min_t)) # collect all angles
        angle_min = np.argmax(abs(np.array(angles))) #select smallest angle
        t_max = target[sing_idxs_min_t[angle_min]] # source[i] and t_max are pair of similar pointing sing matrices
        corresp_idx_t[i] = sing_idxs_min_t[angle_min]

    if with_iso and iso_angle > 0.8:
        source_sing = np.zeros((num+1,3,3))
        target_sing = np.zeros((num+1,3,3))
    else:
        source_sing = np.zeros((num,3,3))
        target_sing = np.zeros((num,3,3))

    for i in np.arange(num):
        source_sing[i] = source[sing_idxs_min_s[i]]
        target_sing[i] = target[corresp_idx_t[i]]

    if with_iso and iso_angle > 0.8:
        source_sing[num] = source[iso_idx_s]
        target_sing[num] = target[iso_idx_t]
        np.append(sing_idxs_min_s,iso_idx_s)
        np.append(corresp_idx_t,iso_idx_t)



    return source_sing, target_sing, sing_idxs_min_s, corresp_idx_t


def find_most_singular_points_diff_dir(source, target, num):
    s=list()
    t=list()
    assert(3*num < source.shape[0])

    path_points_t=list()
    path_points_s=list()
    path_idx_t=list()
    path_idx_s=list()

    print("SING2SING with %i points" %(num))

    # find most singular samples in s
    sing_idxs=list()
    for i in np.arange(source.shape[0]):
        w, _ = np.linalg.eig(source[i])
        sing_idxs.append(max(w)/min(w)) # if ratio very small then singular

    sing_idxs_min_s = np.array(sing_idxs).argsort()[-3*num:]


    # find most singular samples in t
    sing_idxs=list()
    for i in np.arange(target.shape[0]):
        w, _ = np.linalg.eig(target[i])
        sing_idxs.append(max(w)/min(w)) # if ratio very small then singular

    sing_idxs_min_t = np.array(sing_idxs).argsort()[-3*num:]

    # find most different pointing samples within s
    diff_matrix=np.ones((3*num,3*num))*999
    for i in np.arange(3*num):
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


    # find most similar directing sample between s and t (dot product near 1)
    corresp_idx_t = np.zeros(num, dtype=int)
    angles_list=list()
    for i in np.arange(num):
        si=sing_idxs_min_s[i]
        vs,ws = np.linalg.eig(source[si])
        w_min_s = ws[:,np.argmin(vs)] # eigvec corresp to smallest eigval
        angles=list()
        for j in sing_idxs_min_t:
            vt,wt = np.linalg.eig(target[j])
            w_min_t = wt[:,np.argmin(vt)] # eigvec corresp to smallest eigval
            angles.append(w_min_s.dot(w_min_t)) # collect all angles for ti corresp to si
        angle_min = np.argmax(abs(np.array(angles))) #select smallest angle for si
        angles_list.append(angle_min)
        t_max = target[sing_idxs_min_t[angle_min]] # source[i] and t_max are pair of similar pointing sing matrices
        corresp_idx_t[i] = sing_idxs_min_t[angle_min]


    angle_list_sorted=(-np.array(angles_list)).argsort() # descending order, biggest element is smallest angle between correspondences
    s_list=list()
    i=0
    while len(s_list) < num :
        idx_s1 = angle_list_sorted[i] # angle to ti is smallest
        idx_s2 = diff_max_i[i] # angle to other si biggest

        if idx_s1 not in s_list:
            s_list.append(idx_s1)
        if idx_s2 not in s_list:
            s_list.append(idx_s2)
        i+=1



    source_sing = np.zeros((num,3,3))
    target_sing = np.zeros((num,3,3))
    s_list=np.array(s_list)
    t_list=list()

    for i in np.arange(num):
        source_sing[i] = source[s_list[i]]
        target_sing[i] = target[corresp_idx_t[i]]
        t_list.append(corresp_idx_t[i])

    assert(s_list.shape[0]==len(t_list)==num)

    return source_sing, target_sing, s_list, np.array(t_list)



def find_singular_geodesic_paths(source, target, num): # number of singular points to select
    s=list()
    t=list()

    path_points_t=list()
    path_points_s=list()
    path_idx_t=list()
    path_idx_s=list()

    print("SING2SING with %i points" %(num))

    # find most singular samples in s
    sing_idxs=list()
    for i in np.arange(source.shape[0]):
        w, _ = np.linalg.eig(source[i])
        #print(w)
        #print(min(w),max(w),max(w)/min(w))
        sing_idxs.append(max(w)/min(w)) # if ratio very small then singular

    sing_idxs_min_s = np.array(sing_idxs).argsort()[-num:]
    #print("sing_idxs_min_s",sing_idxs_min_s)


    # find most singular samples in t
    sing_idxs=list()
    for i in np.arange(target.shape[0]):
        w, _ = np.linalg.eig(target[i])
        sing_idxs.append(min(w)/max(w)) # if ratio very small then singular

    sing_idxs_min_t = np.array(sing_idxs).argsort()[:num]
    #print("sing_idxs_min_t", sing_idxs_min_t)


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
    #print(corresp_idx_t)

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
    #print(diff_matrix)
    #print("diffmax ",diff_max_i)



    # create paths for each pair (if not already in othetr direction done)
    paths_done=list()

    path_points_si=list()
    path_points_ti=list()
    path_idx_si=list()
    path_idx_ti=list()
    for n in np.arange(num):
        print("---")
        #print(sing_idxs_min_s[n],sing_idxs_min_s[diff_max_i[n]])

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

        # logeuc as distance
        if sing_idxs_min_s[n] == sing_idxs_min_s[diff_max_i[n]]: # as same as bs
            print("a_s same as b_s")
            path_points_si.append(b_s)
            path_idx_si.append(sing_idxs_min_s[diff_max_i[n]])
        else:
            print("USING LOG EUC IN SING2SING")
            path_points_si, path_idx_si = find_singular_geodesic_path_logeuc(source, a_s, b_s, sing_idxs_min_s[n])
            #print("USING SING ANGLE IN SING2SING")
            #path_points_si, path_idx_si = find_singular_geodesic_path_angle(source, a_s, b_s, sing_idxs_min_s[n])

        if corresp_idx_t[n] == corresp_idx_t[diff_max_i[n]]: # at same as bt
            print("a_t same as b_t")
            path_points_ti.append(b_t)
            path_idx_ti.append(corresp_idx_t[diff_max_i[n]])
        else:
            #path_points_ti, path_idx_ti = find_singular_geodesic_path_angle(target, a_t, b_t, corresp_idx_t[n])
            path_points_ti, path_idx_ti = find_singular_geodesic_path_logeuc(target, a_t, b_t, corresp_idx_t[n])
        

        # find nearest neighbors
        source_tmp = np.array(path_points_si)
        target_tmp = np.array(path_points_ti)

        if source_tmp.shape[0] >= target_tmp.shape[0]:
            # if more target points than source points
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

        else:
            # if more target points than source points
            nns = np.zeros(target_tmp.shape[0], dtype=np.int)  # for each target search smallest source

            dst_sum = 0
            for i in np.arange(target_tmp.shape[0]):
                if i== 0: # map a_s to a_t
                    nns[i]=0
                if i== target_tmp.shape[0]-1: # map a_t to b_t
                    nns[i] = source_tmp.shape[0]-1
                else:
                    s=target_tmp[i]
                    dist_min = 9999
                    idx_s = 0
                    for t in source_tmp:
                        ds = distance_riemann(t, s)
                        if ds < dist_min:
                            dist_min = ds
                            nns[i] = int(idx_s)
                        idx_s += 1
                    dst_sum += dist_min
            mean_dist = float(dst_sum / source_tmp.shape[0])

            source_rearranged = np.zeros((target_tmp.shape[0],3,3))

            for i in np.arange(source_rearranged.shape[0]):
                source_rearranged[i] = source_tmp[nns[i]]


        for t in target_rearranged:
            path_points_t.append(t)

        for s in source_tmp:
            path_points_s.append(s)

        for t in path_idx_ti:
            path_idx_t.append(t)

        for s in path_idx_si:
            path_idx_s.append(s)

        print("path_t: %i, path_s: %i, idx_s: %i, idx_t: %i" %(len(path_points_t), len(path_points_s), len(path_idx_s), len(path_idx_t)))


    print("TOTAL LENGHTS OF SING2SING PATHS")
    print(len(path_points_s))
    print(len(path_points_t))
    path_points_t=np.array(path_points_t)
    path_points_s=np.array(path_points_s)
    return path_points_s, path_points_t, np.array(path_idx_s), np.array(path_idx_t)