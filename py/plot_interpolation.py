import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import glob
import pandas as pd
from numpy import genfromtxt
import numpy as np
from scipy import linalg
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

# visualize the interpolation


data_path = "/home/nnrthmr/CLionProjects/ma_thesis/data/demos/rhuman_luis/data/cut_optimal/exp2_task5_joints.csv"
data_path2 = "/home/nnrthmr/CLionProjects/ma_thesis/data/demos/rhuman_luis/data/cut_optimal/interpolated/exp2_task5_joints.csv"


data = pd.read_csv(data_path, header=None,sep=",")
data2 = pd.read_csv(data_path2,sep=",")
print(data.head())
print(data2.head())

xdata= np.array(data[0])
ydata= np.array(data[4])
y2data= np.array(data[5])
x2data= np.array(data2['s'])
zdata= np.array(data2['joint4'])
z2data= np.array(data2['joint5'])


plt.figure()
plt.scatter(xdata, ydata, c='grey', alpha=0.4)
plt.scatter(x2data, zdata, c='red', alpha=0.4)
plt.scatter(xdata, y2data, c='grey', alpha=0.4)
plt.scatter(x2data, z2data, c='green', alpha=0.4)
plt.show()
