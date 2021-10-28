import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pandas as pd

colors=['green', 'blue', 'orange', 'red', 'purple']

def get_cov_ellipsoid(cov, mu=np.zeros((3)), nstd=3):
    """
    Return the 3d points representing the covariance matrix
    cov centred at mu and scaled by the factor nstd.
    Plot on your favourite 3d axis. 
    Example 1:  ax.plot_wireframe(X,Y,Z,alpha=0.1)
    Example 2:  ax.plot_surface(X,Y,Z,alpha=0.1)
    """
    assert cov.shape==(3,3)

    # Find and sort eigenvalues to correspond to the covariance matrix
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.sum(cov,axis=0).argsort()
    eigvals_temp = eigvals[idx]
    idx = eigvals_temp.argsort()
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:,idx]

    # Set of all spherical angles to draw our ellipsoid
    n_points = 100
    theta = np.linspace(0, 2*np.pi, n_points)
    phi = np.linspace(0, np.pi, n_points)

    # Width, height and depth of ellipsoid
    rx, ry, rz = nstd * np.sqrt(eigvals)

    # Get the xyz points for plotting
    # Cartesian coordinates that correspond to the spherical angles:
    X = rx * np.outer(np.cos(theta), np.sin(phi))
    Y = ry * np.outer(np.sin(theta), np.sin(phi))
    Z = rz * np.outer(np.ones_like(theta), np.cos(phi))

    # Rotate ellipsoid for off axis alignment
    old_shape = X.shape
    # Flatten to vectorise rotation
    X,Y,Z = X.flatten(), Y.flatten(), Z.flatten()
    X,Y,Z = np.matmul(eigvecs, np.array([X,Y,Z]))
    X,Y,Z = X.reshape(old_shape), Y.reshape(old_shape), Z.reshape(old_shape)
   
    # Add in offsets for the mean
    X = X + mu[0]
    Y = Y + mu[1]
    Z = Z + mu[2]
    
    return X,Y,Z

fig = plt.figure()
ax = plt.axes(projection='3d')
plt.title('Demonstrations')


for i in np.arange(4):
    #data_path = "../data/demos/EEpos_data_trial_"+str(i)+".csv"
    data_path = "/home/nnrthmr/Desktop/master-thesis/vrep/vrep_franka_promps/py_scripts/data/sphere/EEpos_data_sphere_trial_"+str(i)+".csv"
    data = pd.read_csv(data_path, sep=",")

    xdata= np.array(data['EE_x'])
    ydata= np.array(data['EE_y'])
    zdata= np.array(data['EE_z'])

    ax.scatter3D(xdata, ydata, zdata, c=colors[i])

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
