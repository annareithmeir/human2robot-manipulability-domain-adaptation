import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pandas as pd
from numpy import genfromtxt
from get_cov_ellipsoid import get_cov_ellipsoid
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['text.usetex'] = True




n_demos=1
n_points=10
scaling_factor = 0.15
plot_every_nth = 5

COLS=['s','x','y','z']


# for i in np.arange(n_demos):
#     #data_path = "../data/demos/EEpos_data_trial_"+str(i)+".csv"
#     #data_path = "/home/nnrthmr/Desktop/master-thesis/vrep/vrep_franka_promps/py_scripts/data/sphere/EEpos_data_sphere_trial_"+str(i)+".csv"
#     data_path = "/home/nnrthmr/CLionProjects/ma_thesis/data/demos/towards_singularities/panda/panda_reach_up_t_interpolated.csv"
#     data = pd.read_csv(data_path, sep=",")
#     data.set_axis(COLS, axis=1, inplace=True)
#     print(data)

#     xdata= np.array(data['x'])
#     ydata= np.array(data['y'])
#     zdata= np.array(data['z'])

#     ax.scatter3D(xdata, ydata, zdata, c=colors[i])


###############################################
### plot demonstration manipulabilities     ###
###############################################


fig = plt.figure(figsize=(40, 15))
fig.subplots_adjust(bottom=-0.15,top=1.2,wspace=0, hspace=0, right=1)
#ax = plt.axes(projection='3d')
ax = fig.gca(projection='3d')
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))


### top view ###
fig2 = plt.figure(figsize=(20, 7))
fig2.subplots_adjust(bottom=-0.15,top=1.2,wspace=0, hspace=0, right=1)
#ax2 = plt.axes(projection='3d')
ax2 = fig2.gca(projection='3d')
ax2.set_facecolor('white')
ax2.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax2.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax2.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

#filename_manip = "../data/demos/tum_tuda/translationManip3d.csv"
#filename_manip = "../data/demos/translationManip3d.csv"
#filename_manip = "/home/nnrthmr/Desktop/master-thesis/vrep/vrep_franka_promps/py_scripts/data/sphere/EEpos_translationmanipulability_sphere_trial_"+str(i)+".csv"
filename_manip = "/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/rhuman/sing_up_60/manipulabilities_40.csv"
filename_manip2 = "/home/nnrthmr/CLionProjects/ma_thesis/data/final_results/naive/sing_up_60/manipulabilities_mapped_naive.csv"

colors=["darkorange","rebeccapurple"]

paths=[filename_manip]

for p in paths:

    data = genfromtxt(p, delimiter=',')
    c=0


    print("DATA SHAPE: ", data.shape)

    alpha=0.3

    manip=list()

    for i in np.arange(data.shape[0]):
      mm=data[i].reshape(3,3)
      manip.append(scaling_factor*mm)

    cnt=0
    for i in np.arange(0,len(manip),plot_every_nth):
      m_i = manip[i]
      w,v = np.linalg.eigh(m_i) # just for very singular cases as in trajectories generated
      w[w<1e-12]=0.001
      m=np.matmul(np.matmul(v, np.diag(w)), v.transpose())
      m_i=m
      X2,Y2,Z2 = get_cov_ellipsoid(m_i, [0,1*cnt,0], 1)
      # ax.plot_surface(X2,Y2,Z2, color=colors[c], alpha=alpha)
      ax.plot_wireframe(X2,Y2,Z2, color=colors[c], alpha=alpha)
      # ax2.plot_surface(X2,Y2,Z2, color=colors[c], alpha=alpha)
      ax2.plot_wireframe(X2,Y2,Z2, color=colors[c], alpha=alpha)
      cnt+=1
    c+=1


scale=np.diag([1, 0.5*cnt, 1, 1.0])
scale=scale*(1.0/scale.max())
scale[3,3]=0.7

def short_proj():
    return np.dot(Axes3D.get_proj(ax), scale)
def short_proj2():
    return np.dot(Axes3D.get_proj(ax2), scale)

ax.get_proj=short_proj
ax.set_box_aspect(aspect = (1,1,1))
ax.view_init(azim=0, elev=0)


plt.ylim(-0.5, 1*len(manip)/plot_every_nth+0.5)
plt.xlim(-0.5, 0.5)
ax.set_zlim(-0.5, 0.5)
#ax.set_xlabel('$x$')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_ylabel('\\textit{y (Front view)}',labelpad=40)
ax.set_zlabel('\\textit{z}')


ax2.get_proj=short_proj2
ax2.set_box_aspect(aspect = (1,1,1))
ax2.view_init(azim=0, elev=90)


#plt.ylim(-0.5, 0.5*len(manip)/plot_every_nth+0.5)
#plt.xlim(-0.5, 0.5)
ax2.set_zlim(-0.5, 0.5)
ax2.set_zticks([])
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_xlabel('$x$')
#ax2.yaxis.tick_left()
#ax2.xaxis.set_ticks_position('both')
ax2.xaxis.set_label_position('bottom')
ax2.set_ylabel('\\textit{y (Top view)}', labelpad=40)

#ax.set_title("Front view")


fig.tight_layout()
fig2.tight_layout()

final_results_path="/home/nnrthmr/CLionProjects/ma_thesis/data/final_images/manipulationFigure"

fig.tight_layout()
fig2.tight_layout()
fig.savefig(final_results_path+"/rhuman.svg")
fig2.savefig(final_results_path+"/panda.svg")


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
