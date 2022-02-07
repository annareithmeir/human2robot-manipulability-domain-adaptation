import numpy as np
import matplotlib.pyplot as plt
from pyriemann.utils.distance import distance_riemann, distance_wasserstein
from rpa.helpers.transfer_learning.utils import mean_riemann, parallel_transport_covariances
from gca import get_gca, log_map, exp_map
from get_cov_ellipsoid import get_cov_ellipsoid, get_volume
from mpl_toolkits.mplot3d import Axes3D
from pyriemann.utils.geodesic import geodesic_riemann
import matplotlib.patches as mpatches
from pyriemann.utils.base import invsqrtm, sqrtm, logm, expm, powm
from numpy import linalg as LA
from get_cov_ellipsoid import get_volume, scale_volume_to_desired_vol, get_logeuclidean_distance
import math


def get_errors(data1, data2):

	mse_icp=0.0
	le_mse_icp=0.0

	for (mi,mj) in zip(data1, data2):
		print("%.3f " %(distance_riemann(mi, mj)))
		mse_icp += distance_riemann(mi, mj)**2
		le_mse_icp += get_logeuclidean_distance(mi, mj)**2

	n_points = data1.shape[0]
	print("MSE (riemann): %.3f " %(mse_icp/n_points))
	print("MSE (LogEuc): %.3f " %(le_mse_icp/n_points))




def plot_ellipsoids(datalist, labellist=None): # nx3x3 format
	colors=['cornflowerblue', 'darkblue', 'mediumorchid', 'plum', 'darkorange']
	scaling_factor=1e-1
	plot_every_nth=10

	fig = plt.figure()
	plt.title("CPD Rigid")
	ax = plt.axes(projection='3d')

	c=0
	mse_icp=0.0

	le_mse_naive=0.0
	le_mse_icp=0.0

	for data in datalist:

		manip=list()

		for i in np.arange(data.shape[0]):
			manip.append(scaling_factor*data[i])

		cnt=0
		for i in np.arange(0,len(manip),plot_every_nth):
			m_i = manip[i]
			X2,Y2,Z2 = get_cov_ellipsoid(m_i, [1*cnt,0,0], 1)
			ax.plot_wireframe(X2,Y2,Z2, color=colors[c], alpha=0.05)
			cnt+=1
		c+=1


	# print errors
	get_errors(datalist[0],datalist[1])

	scale=np.diag([cnt, 1, 1, 1.0])
	scale=scale*(1.0/scale.max())
	scale[3,3]=0.7
	def short_proj():
	  return np.dot(Axes3D.get_proj(ax), scale)

	ax.get_proj=short_proj
	ax.set_box_aspect(aspect = (1,1,1))

	if labellist is not None:
		blue_patch = mpatches.Patch(color='cornflowerblue', label=labellist[0])
		red_patch = mpatches.Patch(color='darkblue', label=labellist[1])
		if len(labellist)==3:
			orange_patch = mpatches.Patch(color='mediumorchid', label=labellist[2])
			plt.legend(handles=[ blue_patch, red_patch, orange_patch])
		else:
			plt.legend(handles=[ blue_patch, red_patch])

	plt.xlim(-0.5, len(manip)/plot_every_nth)
	plt.ylim(-0.5, 0.5)
	ax.set_zlim(-0.5, 0.5)
	plt.show()






def SPD_to_6d():
	return 0

def SPD_from_6d():
	return 0

# main and minor axis and volume
def SPD_to_8d(data):
	data_vec=np.zeros((data.shape[0], 8))

	for i in np.arange(data.shape[0]):
		vs,ws = np.linalg.eig(data[i])
		data_vec[i,0:3] = ws[:,np.argmin(vs)] # eigvec corresp to smallest eigval -> minor axis
		data_vec[i,3:6] = ws[:,np.argmax(vs)] # eigvec corresp to smallest eigval -> major axis
		data_vec[i,6]= math.sqrt(np.max(vs)) / math.sqrt(np.min(vs)) # ratio of lengths
		data_vec[i,7] = get_volume(data[i])

	return data_vec

# main/minor axis/volume and middle of axes as secondary axis
def SPD_from_8d(data_vectorized):
	data=np.zeros((data_vectorized.shape[0], 3,3))

	for i in np.arange(data_vectorized.shape[0]):
		min_ax = data_vectorized[i,0:3] # eigvec corresp to smallest eigval -> minor axis normalized
		max_ax = data_vectorized[i,3:6] # eigvec corresp to smallest eigval -> major axis normalized
		ratio = data_vectorized[i,6]
		vol = data_vectorized[i,7]

		sec_ax = np.cross(min_ax, max_ax) # find secondary axis 
		sec_ax = sec_ax / np.linalg.norm(sec_ax) # normalize to unit vector
		min_ax = min_ax / np.linalg.norm(min_ax) # normalize to unit vector
		max_ax = max_ax / np.linalg.norm(max_ax) # normalize to unit vector
		#print(np.linalg.norm(sec_ax),np.linalg.norm(min_ax),np.linalg.norm(max_ax))

		#assert(np.linalg.norm(sec_ax)==1)
		#assert(np.linalg.norm(min_ax)==1)
		#assert(np.linalg.norm(max_ax)==1)
		

		U = np.zeros((3,3))
		V = np.zeros((3,3))
		U[:,0]=min_ax
		U[:,1]=sec_ax
		U[:,2]=max_ax
		V[0,0]=1 # eigs = lenghts^2, set secondary axis length inbetween major and minor
		V[1,1]=(ratio*0.5)**2
		V[2,2]=ratio**2
		A =  U @ V @ U.transpose() # build ellipsoid from eigendecomposition

		data[i] = scale_volume_to_desired_vol(A, abs(vol)) # scale to correct volume


		if i==32:
			print(np.linalg.norm(sec_ax),np.linalg.norm(min_ax),np.linalg.norm(max_ax))
			print(min_ax)
			print(max_ax)
			print(sec_ax)
			print(A)
			print(vol)
			print(data[i])
			print("----")

	return data

def perform_8d_trafo(data_new, s, R, t):
	#data_new=s*data_new*R'+repmat(t',[M 1]);

	print(R.shape)
	print(t.shape)
	print(s.shape, s)
	for i in np.arange(data_new.shape[0]):
		data_new[i] = s* data_new[i] @ R.transpose() + t.transpose()
	return data_new

def plot_8d_original_and_mapped():
	data_groundtruth = np.genfromtxt("/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/toy_data/100/manipulabilities.csv", delimiter=',')
	data_groundtruth = np.reshape(data_groundtruth,(data_groundtruth.shape[0],3,3))
	#data_groundtruth2 = np.genfromtxt("/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/toy_data/100/8d/manipulabilities.csv", delimiter=',')
	data_mapped = np.genfromtxt("/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/toy_data/100/8d/manipulabilities_mapped.csv", delimiter=',')

	# data_groundtruth = np.genfromtxt("/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/toy_data/10_new/manipulabilities_interpolated_groundtruth.csv", delimiter=',')
	# data_groundtruth = data_groundtruth[:,1:]
	# data_groundtruth = np.reshape(data_groundtruth,(data_groundtruth.shape[0],3,3))
	# data_mapped = np.genfromtxt("/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/toy_data/10_new/8d/manipulabilities_mapped.csv", delimiter=',')

	data_re_mapped = SPD_from_8d(data_mapped)
	plot_ellipsoids([data_groundtruth, data_re_mapped],["groundtruth", "mapped"])

	# data_re_gt2 = SPD_from_8d(data_groundtruth2)
	# plot_ellipsoids([data_groundtruth, data_re_gt2, data_re_mapped])



# eigenvalues as 3d points
def SPD_to_3d(data):
	eigs=np.zeros((data.shape[0],3))

	for i in np.arange(data.shape[0]):
		w = LA.eigvalsh(data[i])
		eigs[i,:] = np.sort(w)
	return eigs


# from eigenvalues as 3d points
def SPD_from_3d():

	return 0



# data_franka = np.genfromtxt("/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/panda/100/manipulabilities.csv", delimiter=',')
# data_franka = np.reshape(data_franka,(data_franka.shape[0],3,3))
# data_v = SPD_to_3d(data_franka)
# # data_v = SPD_to_8d(data_franka)
# np.savetxt("/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/panda/100/3d/manipulabilities.csv", data_v, delimiter=",")

# data_2 = np.genfromtxt("/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/rhuman/100/manipulabilities.csv", delimiter=',')
# data_2 = np.reshape(data_2,(data_2.shape[0],3,3))
# data_v2 = SPD_to_3d(data_2)
# # data_v2 = SPD_to_8d(data_2)
# np.savetxt("/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/rhuman/100/3d/manipulabilities.csv", data_v2, delimiter=",")

# data_3 = np.genfromtxt("/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/toy_data/100/manipulabilities.csv", delimiter=',')
# data_3 = np.reshape(data_3,(data_3.shape[0],3,3))
# data_v3 = SPD_to_3d(data_3)
# # data_v3 = SPD_to_8d(data_3)
# np.savetxt("/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/toy_data/100/3d/manipulabilities.csv", data_v3, delimiter=",")


plot_8d_original_and_mapped()



# data_new = np.genfromtxt("/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/panda/10_new/manipulabilities_interpolated.csv", delimiter=',')
# data_new=data_new[:,1:]
# data_new = np.reshape(data_new,(data_new.shape[0],3,3))
# R = np.genfromtxt("/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/panda/100/8d/R_cpd_panda_to_toy_data.txt", delimiter=',')
# t = np.genfromtxt("/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/panda/100/8d/t_cpd_panda_to_toy_data.txt", delimiter=',')
# s = np.genfromtxt("/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/panda/100/8d/s_cpd_panda_to_toy_data.txt", delimiter=',')
# data_new_8d = SPD_to_8d(data_new)
# data_new_mapped = perform_8d_trafo(data_new_8d, s,R,t)
# np.savetxt("/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/toy_data/10_new/8d/manipulabilities_mapped.csv", data_new_mapped, delimiter=",")
# plot_8d_original_and_mapped()











#data_re= SPD_from_8d(data_v)
#plot_ellipsoids([data_franka[:10], data_re])

#data_v=SPD_to_3d(data_franka)