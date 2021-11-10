import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pandas as pd
from numpy import genfromtxt
import get_cov_ellipsoid

colors=['green', 'blue', 'orange', 'red', 'purple']


fig = plt.figure()
ax = plt.axes(projection='3d')
plt.title('Demonstrations')

n_demos=1
n_points=400
scaling_factor = 1e-1
plot_every_nth = 30


for i in np.arange(n_demos):
    #data_path = "../data/demos/EEpos_data_trial_"+str(i)+".csv"
    data_path = "/home/nnrthmr/Desktop/master-thesis/vrep/vrep_franka_promps/py_scripts/data/sphere/EEpos_data_sphere_trial_"+str(i)+".csv"
    data = pd.read_csv(data_path, sep=",")

    xdata= np.array(data['EE_x'])
    ydata= np.array(data['EE_y'])
    zdata= np.array(data['EE_z'])

    ax.scatter3D(xdata, ydata, zdata, c=colors[i])


###############################################
### plot demonstration manipulabilities     ###
###############################################

#filename_manip = "../data/demos/tum_tuda/translationManip3d.csv"
#filename_manip = "../data/demos/translationManip3d.csv"
filename_manip = "/home/nnrthmr/Desktop/master-thesis/vrep/vrep_franka_promps/py_scripts/data/sphere/EEpos_translationmanipulability_sphere_trial_"+str(i)+".csv"
manip_tmp = genfromtxt(filename_manip, delimiter=',')
manip_tmp=manip_tmp[1:,:]
print(manip_tmp)
manip=list()


for i in np.arange(0, manip_tmp.shape[0]):
    manip.append(scaling_factor*manip_tmp[i,2:].reshape(3,3))

manip=manip[:n_demos*n_points]

for i in np.arange(0,len(manip),plot_every_nth):
    m_i = manip[i]
    X2,Y2,Z2 = get_cov_ellipsoid(m_i, [xdata[i],ydata[i],zdata[i]], 1)
    ax.plot_wireframe(X2,Y2,Z2, color='grey', alpha=0.05)



ax.set_zlim(0, 1)
plt.xlim(-1 ,1)
plt.ylim(-1, 1)
plt.show()

###############################################
# human arm motion data                       #
###############################################
'''
    First 12 values are the joint values
    values 13-15 are the wrist position
    values 16-18 are the wrist orientation in xyz
    values 19-54 are the wrist position jacobian (first row, then second row, then third row)
    values 55-90 are the wrist orientation jacobian (in the same format as above)

data_path = "/home/nnrthmr/CLionProjects/ma_thesis/data/demos/human_arm/humanArmMotionOutput.txt"
data = pd.read_csv(data_path, sep=" ", skiprows=1, header=None)

xdata= np.array(data[12])
ydata= np.array(data[13])
zdata= np.array(data[14])
print(data[12:15])
manip=list()

for i in np.arange(0, 8219):
    tmp=np.array(data.iloc[i][18:54]).reshape((3,12))
    manip.append(0.000001*np.matmul(tmp,np.transpose(tmp)))

for i in np.arange(0,len(manip),150):
    m_i = manip[i]
    X2,Y2,Z2 = get_cov_ellipsoid(m_i, [xdata[i],ydata[i],zdata[i]], 1)
    ax.plot_wireframe(X2,Y2,Z2, color='grey', alpha=0.05)

ax.scatter3D(xdata, ydata, zdata, c=colors[0])

plt.show()
'''
