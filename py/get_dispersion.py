import numpy as np
import matplotlib.pyplot as plt
from pyriemann.utils.distance import distance_riemann, distance_wasserstein
from pyriemann.utils.base import invsqrtm, sqrtm, logm, expm, powm
from rpa.helpers.transfer_learning.utils import mean_riemann, parallel_transport_covariances

from numpy import linalg as LA

from scipy.linalg import eigh
import math



l=["18","2", "4"]


for i in l:
    data_path="/home/nnrthmr/CLionProjects/ma_thesis/data/final_results/2dof-2dofvertical/PT/500/validation/manipulabilities_"+i+"/manipulabilities_mapped_icp.csv"
    data = np.genfromtxt(data_path, delimiter=',')
    data = data.reshape((data.shape[0], 3, 3))

    mean_data=mean_riemann(data)
    data = np.stack([np.dot(invsqrtm(mean_data), np.dot(ti, invsqrtm(mean_data))) for ti in data])  
    disp = np.sum([distance_riemann(covi, np.eye(3)) ** 2 for covi in data]) / len(data)

    print(i)
    print(disp)


