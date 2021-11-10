import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import glob
from numpy import genfromtxt
import numpy as np
from scipy import linalg
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import pandas as pd
from get_cov_ellipsoid import get_cov_ellipsoid




experiment_name="cut_userchoice"
proband=4


plot_trajectories_all = False
plot_manipulabilities_selected = False
plot_GMM = 1
plot_time_traj = 0
plot_time_loop = False

###########################
### plot demonstrations ### # 3x4 grid of all trials demonstrated by all probands
###########################

colors=['green', 'blue', 'orange', 'red', 'purple', 'blue', 'gold', 'pink','grey', 'coral']
experiments=['cut_optimal', 'cut_random', 'cut_userchoice']
#experiments=['drill_optimal', 'drill_random', 'drill_userchoice']


######  trajectories  #######

if plot_trajectories_all:
    fig = plt.figure(1, figsize=plt.figaspect(0.3))
    plt.suptitle('Demonstrations rhuman experiments')
    for ex in np.arange(len(experiments)):
        experiment_name=experiments[ex]
        for proband in np.arange(2,6):
            ax = fig.add_subplot(3, 4, proband-1+4*ex, projection='3d')
            ax.title.set_text(experiment_name+": "+str(proband))
            files = glob.glob("/home/nnrthmr/Desktop/RHuMAn-arm-model/data/"+experiment_name+"/exp"+str(proband)+"*_t.csv")

            with open("/home/nnrthmr/Desktop/RHuMAn-arm-model/data/"+experiment_name+"/agg/"+str(proband)+"/info.txt") as f:
                info = f.readline()
            nPoints= int(info.strip().split(' ')[0])
            #print('nPoints =', nPoints)


            for i in np.arange(len(files)):
                data_path = files[i]
                data = pd.read_csv(data_path, sep=",", names=['EE_x','EE_y','EE_z'])

                xdata= np.array(data['EE_x'])[:nPoints]
                ydata= np.array(data['EE_y'])[:nPoints]
                zdata= np.array(data['EE_z'])[:nPoints]
                ax.plot(xdata, ydata, zdata, c=colors[i], alpha=0.5)

    fig.tight_layout()
    #plt.show()


####  manipulabilities  #### # only from one selected file


if plot_manipulabilities_selected:
    fig = plt.figure(2, figsize=plt.figaspect(0.3))
    plt.suptitle('Demonstrations rhuman experiments')
    scaling_factor=1e-3
    plot_every_nth = 120

    with open("/home/nnrthmr/Desktop/RHuMAn-arm-model/data/"+experiment_name+"/agg/"+str(proband)+"/info.txt") as f:
        info = f.readline()
    nPoints= int(info.strip().split(' ')[0])
    print('nPoints =', nPoints)

    ax = plt.axes(projection='3d')

    experiment_name="cut_optimal"
    files_m = sorted(glob.glob("/home/nnrthmr/Desktop/RHuMAn-arm-model/data/"+experiment_name+"/exp" + str(proband)+"*_m.csv"))
    files_t = sorted(glob.glob("/home/nnrthmr/Desktop/RHuMAn-arm-model/data/"+experiment_name+"/exp"+ str(proband) +"*_t.csv"))

    data = pd.read_csv(files_t[0], sep=",", names=['EE_x','EE_y','EE_z'])
    xdata= np.array(data['EE_x'])[:nPoints]
    ydata= np.array(data['EE_y'])[:nPoints]
    zdata= np.array(data['EE_z'])[:nPoints]
    ax.plot(xdata, ydata, zdata, c='black', alpha=0.5)

    filename_manip = files_m[0]
    manip_tmp = genfromtxt(filename_manip, delimiter=',')
    manip=list()

    for i in np.arange(0, manip_tmp.shape[0]):
        manip.append(scaling_factor*manip_tmp[i,1:].reshape(3,3))

    #for i in np.arange(0,len(manip),plot_every_nth):
    for i in np.arange(0,970,plot_every_nth):
        m_i = manip[i]
        X2,Y2,Z2 = get_cov_ellipsoid(m_i, [xdata[i],ydata[i],zdata[i]], 1)
        ax.plot_wireframe(X2,Y2,Z2, color='grey', alpha=0.05)

    fig.tight_layout()
    #plt.show()


### GMM results ###

if plot_GMM:
    fig = plt.figure(3, figsize=plt.figaspect(0.3))
    plt.suptitle('Demonstrations rhuman experiments')
    plot_every_nth=500
    scaling_factor=1e-4

    with open("/home/nnrthmr/Desktop/RHuMAn-arm-model/data/"+experiment_name+"/agg/"+str(proband)+"/info.txt") as f:
        info = f.readline()
    nPoints= int(info.strip().split(' ')[0])
    print(experiment_name, ' ', proband,' nPoints =', nPoints)
    nPoints=int(nPoints)

    ax = plt.axes(projection='3d')
    ax.title.set_text(experiment_name+', proband '+str(proband))
    files = sorted(glob.glob("/home/nnrthmr/Desktop/RHuMAn-arm-model/data/"+experiment_name+"/exp"+str(proband)+"*_t.csv"))
    files_m = sorted(glob.glob("/home/nnrthmr/Desktop/RHuMAn-arm-model/data/"+experiment_name+"/exp"+str(proband)+"*_m.csv"))

    for i in np.arange(len(files)):
        #trajectories demos
        data_path = files[i]
        data = pd.read_csv(data_path, sep=",", names=['EE_x','EE_y','EE_z'])

        xdata= np.array(data['EE_x'])[:nPoints]
        ydata= np.array(data['EE_y'])[:nPoints]
        zdata= np.array(data['EE_z'])[:nPoints]
        ax.plot(xdata, ydata, zdata, c=colors[i], alpha=0.25)

        
        # manips demos
        filename_manip = files_m[i]
        manip_tmp = genfromtxt(filename_manip, delimiter=',')
        manip=list()

        for i in np.arange(0, manip_tmp.shape[0]):
            manip.append(scaling_factor*manip_tmp[i,1:].reshape(3,3))

        for i in np.arange(0,nPoints,plot_every_nth):
            m_i = manip[i]
            X2,Y2,Z2 = get_cov_ellipsoid(m_i, [xdata[i],ydata[i],zdata[i]], 1)
            ax.plot_wireframe(X2,Y2,Z2, color='grey', alpha=0.05)
        

    #trajectories GMM result
    mu_tmp = genfromtxt("/home/nnrthmr/CLionProjects/ma_thesis/data/results/rhuman/"+experiment_name+"/"+str(proband)+"/xd.csv", delimiter=',')
    sigma_tmp = genfromtxt("/home/nnrthmr/CLionProjects/ma_thesis/data/results/rhuman/"+experiment_name+"/"+str(proband)+"/xhat.csv", delimiter=',')

    sigma_tmp = scaling_factor*sigma_tmp
    mu_tmp=mu_tmp

    mu=list()
    sigma=list()

    for i in np.arange(mu_tmp.shape[1]):
        mu.append(mu_tmp[:,i])
        sigma.append(sigma_tmp[i,:].reshape(3,3))

    for i in np.arange(len(mu)):
        mu_i = mu[i]
        ax.scatter(mu_i[0],mu_i[1],mu_i[2], color='black', alpha=0.4, s=1)

    
    for i in np.arange(0,len(mu),plot_every_nth):
        mu_i = mu[i]
        sigma_i=sigma[i]
        X2,Y2,Z2 = get_cov_ellipsoid(sigma_i, mu_i, 1)
        ax.plot_wireframe(X2,Y2,Z2, color='blue', alpha=0.1)
    
    fig.tight_layout()
    #plt.show()

### Evolution of manipulabilities throughout trajectories
    
if plot_time_traj:
    fig = plt.figure(4, figsize=plt.figaspect(0.3))
    plt.suptitle('Demonstrations rhuman experiments')
    n_demos=1
    scaling_factor=1e-2
    plot_every_nth = 500

    with open("/home/nnrthmr/Desktop/RHuMAn-arm-model/data/"+experiment_name+"/agg/"+str(proband)+"/info.txt") as f:
        info = f.readline()
    nPoints= int(info.strip().split(' ')[0])
    print(experiment_name, ' ', proband,' nPoints =', nPoints)
    nPoints=int(nPoints)

    plot_time_demos = True
    if plot_time_demos:
        ax = plt.axes(projection='3d')
        plt.title('Manipulabilities over time')
        files_m = sorted(glob.glob("/home/nnrthmr/Desktop/RHuMAn-arm-model/data/"+experiment_name+"/exp"+str(proband)+"*_m.csv"))
        
        for f in np.arange(n_demos):
            demo_tmp = genfromtxt(files_m[f], delimiter=',')
            demo_tmp=demo_tmp[:,:]
            demo=list()


            for i in np.arange(0, nPoints):
                demo.append(scaling_factor*demo_tmp[i,1:].reshape(3,3))


            cnt=0
            for i in np.arange(0,len(demo),plot_every_nth):
                m_i = demo[i]
                X2,Y2,Z2 = get_cov_ellipsoid(m_i, [cnt,0,0], 1)
                ax.plot_wireframe(X2,Y2,Z2, color='green', alpha=0.05)
                cnt+=1

    plot_time_learned = True
    if plot_time_learned:
        file_m = "/home/nnrthmr/CLionProjects/ma_thesis/data/results/rhuman/"+experiment_name+"/"+str(proband)+"/xhat.csv"
        
        demo_tmp = genfromtxt(file_m, delimiter=',')
        demo_tmp=demo_tmp[:,:]
        demo=list()


        for i in np.arange(0, nPoints):
            demo.append(scaling_factor*demo_tmp[i,:].reshape(3,3))


        cnt=0
        for i in np.arange(0,len(demo),plot_every_nth):
            m_i = demo[i]
            X2,Y2,Z2 = get_cov_ellipsoid(m_i, [cnt,0,0], 1)
            ax.plot_wireframe(X2,Y2,Z2, color='blue', alpha=0.05)
            cnt+=1

    plot_time_controlled = False
    if plot_time_controlled:
        filename_controlled="/home/nnrthmr/CLionProjects/ma_thesis/data/tracking/rhuman/"+experiment_name+"/"+str(proband)+"/xhat.csv"
        controlled_tmp = genfromtxt(filename_controlled, delimiter=',')
        controlled_tmp = scaling_factor*controlled_tmp

        controlled=list()
        for i in np.arange(n_points):
            controlled.append(controlled_tmp[i,:].reshape(3,3))

        cnt=0
        for i in np.arange(0,n_points,plot_every_nth):
            controlled_i=controlled[i]
            X2,Y2,Z2 = get_cov_ellipsoid(controlled_i, [4*cnt,0,0], 1)
            ax.plot_wireframe(X2,Y2,Z2, color='blue', alpha=0.1)
            cnt+=1

    fig.tight_layout()

    # scale=np.diag([10, 1, 1, 1.0])
    # scale=scale*(1.0/scale.max())
    # scale[3,3]=1.0

    # def short_proj():
    #   return np.dot(Axes3D.get_proj(ax), scale)

    # ax.get_proj=short_proj
    #plt.show()


### Evolution of manipulabilities throughout one single control loop
    
if plot_time_loop:
    fig = plt.figure(5, figsize=plt.figaspect(0.3))
    plt.suptitle('Demonstrations rhuman experiments')
    n_demos=2
    scaling_factor=1e-1
    plot_every_nth = 1000

    with open("/home/nnrthmr/Desktop/RHuMAn-arm-model/data/"+experiment_name+"/agg/"+str(proband)+"/info.txt") as f:
        info = f.readline()
    nPoints= int(info.strip().split(' ')[0])
    print(experiment_name, ' ', proband,' nPoints =', nPoints)
    nPoints=int(nPoints)

    # plot_time_demos = True
    # if plot_time_demos:
    #     ax = plt.axes(projection='3d')
    #     plt.title('Manipulabilities over time')
    #     files_m = sorted(glob.glob("/home/nnrthmr/Desktop/RHuMAn-arm-model/data/"+experiment_name+"/exp"+str(proband)+"*_m.csv"))
        
    #     for f in np.arange(n_demos):
    #         demo_tmp = genfromtxt(files_m[f], delimiter=',')
    #         demo_tmp=demo_tmp[:,:]
    #         demo=list()


    #         for i in np.arange(0, nPoints):
    #             demo.append(scaling_factor*demo_tmp[i,1:].reshape(3,3))


    #         cnt=0
    #         for i in np.arange(0,len(demo),plot_every_nth):
    #             m_i = demo[i]
    #             X2,Y2,Z2 = get_cov_ellipsoid(m_i, [cnt,0,0], 1)
    #             ax.plot_wireframe(X2,Y2,Z2, color='grey', alpha=0.05)
    #             cnt+=1

    plot_time_learned = True
    if plot_time_learned:
        file_m = "/home/nnrthmr/CLionProjects/ma_thesis/data/results/rhuman/"+experiment_name+"/"+str(proband)+"/xhat.csv"
        
        demo_tmp = genfromtxt(file_m, delimiter=',')
        demo_tmp=demo_tmp[:,:]
        demo=list()

        for i in np.arange(0, nPoints):
            demo.append(scaling_factor*demo_tmp[0,:].reshape(3,3))

        cnt=0
        for i in np.arange(0,len(demo),plot_every_nth):
            m_i = demo[i]
            X2,Y2,Z2 = get_cov_ellipsoid(m_i, [cnt,0,0], 1)
            ax.plot_wireframe(X2,Y2,Z2, color='blue', alpha=0.05)
            cnt+=1

    plot_time_controlled = True
    if plot_time_controlled:
        filename_controlled="/home/nnrthmr/CLionProjects/ma_thesis/data/tracking/rhuman/"+experiment_name+"/"+str(proband)+"/loopManipulabilities.csv"
        controlled_tmp = genfromtxt(filename_controlled, delimiter=',')
        controlled_tmp = scaling_factor*controlled_tmp

        controlled=list()
        for i in np.arange(n_points):
            controlled.append(controlled_tmp[i,:].reshape(3,3))

        cnt=0
        for i in np.arange(0,n_points,plot_every_nth):
            controlled_i=controlled[i]
            X2,Y2,Z2 = get_cov_ellipsoid(controlled_i, [1*cnt,0,0], 1)
            ax.plot_wireframe(X2,Y2,Z2, color='blue', alpha=0.1)
            cnt+=1



    scale=np.diag([1, 1, 1, 1.0])
    scale=scale*(1.0/scale.max())
    scale[3,3]=1.0

    def short_proj():
      return np.dot(Axes3D.get_proj(ax), scale)

    ax.get_proj=short_proj

    blue_patch = mpatches.Patch(color='blue', label='Controlled')
    grey_patch = mpatches.Patch(color='orange', label='Demonstrated')
    red_patch = mpatches.Patch(color='red', label='Error')

    plt.legend(handles=[ blue_patch, grey_patch, red_patch])
    plt.xlim(-0.5, demo_tmp.shape[0]/plot_every_nth)
    plt.ylim(-1, 1)
    ax.set_zlim(-1, 1)

    fig.tight_layout()

plt.show()
    







