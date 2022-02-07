import numpy as np
import matplotlib.pyplot as plt
from rpa.helpers.transfer_learning.utils import mean_riemann, parallel_transport_covariances
from pyriemann.utils.tangentspace import tangent_space, untangent_space

def geodesic_distance(x,y):
    dot_product = np.sum(x*y)
    mag_x = np.linalg.norm(x)
    mag_y = np.linalg.norm(y)
    cosine = dot_product/(mag_x*mag_y)
    if cosine>1: cosine = 1
    if cosine<-1: cosine = -1
    return np.arccos(cosine)


# def log_map(x,y):
#     d = geodesic_distance(x,y)
#     temp = y - np.sum(x*y) * x
#     if np.linalg.norm(temp) != 0:
#         mapped_value = d * (temp/np.linalg.norm(temp))
#     else:
#         mapped_value = np.array([0.0,0.0,0.0])
#     return mapped_value

# def exp_map(p,v):
#     mag_v = np.linalg.norm(v)
#     if mag_v == 0:
#         return p
#     v_normalized = v/mag_v
#     mapped_value = p * np.cos(mag_v) + v_normalized * np.sin(mag_v)
#     return mapped_value

# checked!
def log_map(X,S):
    d,v = np.linalg.eig(np.dot(np.linalg.inv(S),X))
    U = S @ v @ np.diag(np.log(d)) @ (np.linalg.inv(v))
    return U

# checked
def exp_map(U, S):
    d,v = np.linalg.eig(np.linalg.inv(S)@ U)
    X = S @ v @ np.diag(np.exp(d)) @ (np.linalg.inv(v))
    return X

def parallel_transport(v,p,q):
    logmap1 = log_map(p,q)
    logmap2 = log_map(q,p)
    if np.linalg.norm(logmap1)!=0 and np.linalg.norm(logmap2)!=0:
        transported_value = v - (np.dot(logmap1 , v)/geodesic_distance(p,q)) * (logmap1+logmap2)
    else:
        transported_value = v
    return transported_value


def get_gca(data1, data2):
    mean1 = mean_riemann(data1)
    mapped_points1 = np.array([log_map(data1[i], mean1) for i in np.arange(data1.shape[0])])
    print(mapped_points1.T.shape)
    principal_vectors1 = np.linalg.svd(mapped_points1.T)[0]
    magnitudes1 = np.linalg.svd(mapped_points1.T)[1]
    print("First dataset")

    print("Principal Vectors = \n",principal_vectors1)
    print()
    print("Magnitude of Principal vectors = \n",magnitudes1)
    print("Magnitude of Principal vectors = \n",np.linalg.norm(magnitudes1, axis=1))

    print("\n\n")

    mean2 = mean_riemann(data2)
    mapped_points2 = np.array([log_map(data2[i], mean2) for i in np.arange(data2.shape[0])])
    principal_vectors2 = np.linalg.svd(mapped_points2.T)[0]
    magnitudes2 = np.linalg.svd(mapped_points2.T)[1]
    print("Second dataset")

    print("Principal Vectors = \n",principal_vectors2)
    print()
    print("Magnitude of Principal vectors = \n",magnitudes2)

    print("Magnitude of Principal vectors = \n",np.linalg.norm(magnitudes2, axis=1))


