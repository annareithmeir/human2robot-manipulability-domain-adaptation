import numpy as np
import matplotlib.pyplot as plt
from pyriemann.utils.distance import distance_riemann, distance_wasserstein
from rpa.helpers.transfer_learning.utils import mean_riemann, parallel_transport_covariances
from gca import get_gca, log_map, exp_map
from get_cov_ellipsoid import get_cov_ellipsoid, get_volume
from mpl_toolkits.mplot3d import Axes3D
from pyriemann.utils.geodesic import geodesic_riemann
from pyriemann.utils.base import invsqrtm, sqrtm, logm, expm, powm
from numpy import linalg as LA

colors=['cornflowerblue', 'darkblue', 'mediumorchid', 'plum', 'darkorange']

def normalize(a):
	return (a - np.min(a))/np.ptp(a)


def plot_ellipsoids(datalist): # nx3x3 format
	scaling_factor=1e-1
	plot_every_nth=1

	fig = plt.figure()
	ax = plt.axes(projection='3d')

	c=0

	for data in datalist:
		manip=list()

		for i in np.arange(0, data.shape[0]):
		    manip.append(scaling_factor*data[i])

		cnt=0
		for i in np.arange(0,len(manip),plot_every_nth):
		    m_i = manip[i]
		    X2,Y2,Z2 = get_cov_ellipsoid(m_i, [1*cnt,0,0], 1)
		    ax.plot_wireframe(X2,Y2,Z2, color=colors[c], alpha=0.05)
		    cnt+=1
		c+=1

	scale=np.diag([cnt, 1, 1, 1.0])
	scale=scale*(1.0/scale.max())
	scale[3,3]=0.7
	def short_proj():
	  return np.dot(Axes3D.get_proj(ax), scale)

	ax.get_proj=short_proj
	ax.set_box_aspect(aspect = (1,1,1))

	plt.xlim(-0.5, len(manip)/plot_every_nth)
	plt.ylim(-0.5, 0.5)
	ax.set_zlim(-0.5, 0.5)
	plt.show()


def basic_analysis():
	data_franka = np.genfromtxt("/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/panda/100/manipulabilities.csv", delimiter=',')
	data_rhuman = np.genfromtxt("/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/rhuman/100/manipulabilities.csv", delimiter=',')
	data_franka = np.reshape(data_franka,(data_franka.shape[0],3,3))
	data_rhuman = np.reshape(data_rhuman,(data_rhuman.shape[0],3,3))

	print("Number of smaples used: %i" %(data_franka.shape[0]))

	# mean analysis
	print(" ")
	print("Mean franka: \n")
	print(mean_riemann(data_franka))
	print("Mean rhuman: \n")
	print(mean_riemann(data_rhuman))
	print("Riemann distance between means: %.3f" %(distance_riemann(mean_riemann(data_rhuman),mean_riemann(data_franka))))


	# cov analysis
	print(" ")

	# dispersion
	disp_franka = np.sum([distance_riemann(covi, np.eye(3)) ** 2 for covi in data_franka]) / data_franka.shape[0]
	disp_rhuman = np.sum([distance_riemann(covi, np.eye(3)) ** 2 for covi in data_rhuman]) / data_rhuman.shape[0]
	print(" ")
	print("Dispersion franka: %.3f" %(disp_franka))
	print("Dispersion rhuman: %.3f" %(disp_rhuman))

	# gca 
	print(" ")
	print("GCA (dataset1=franka, dataset2=rhuman)")
	get_gca(data_franka, data_rhuman)


def shape_change_analysis():
	data_franka = np.genfromtxt("/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/panda/10_new/manipulabilities_interpolated.csv", delimiter=',')
	data_franka=data_franka[:,1:]
	data_franka = np.reshape(data_franka,(data_franka.shape[0],3,3))

	mean_data=mean_riemann(data_franka)
	print("Dispersion around mean: %.3f" %(np.sum([distance_riemann(covi, np.eye(3)) ** 2 for covi in data_franka]) / len(data_franka)))

	# move mean to id
	data_mean_centered = np.stack([np.dot(invsqrtm(mean_data), np.dot(ti, invsqrtm(mean_data))) for ti in data_franka]) 

	# move mean a little bit in random direction
	#move_to=np.eye(3)
	move_to=data_franka[0]
	mean_shifted=geodesic_riemann(mean_data, move_to,0.1) 
	data_mean_shifted1 = np.stack([np.dot(sqrtm(mean_shifted), np.dot(ti, sqrtm(mean_shifted))) for ti in data_mean_centered]) 
	
	mean_shifted=geodesic_riemann(mean_data, move_to,0.5) 
	data_mean_shifted2 = np.stack([np.dot(sqrtm(mean_shifted), np.dot(ti, sqrtm(mean_shifted))) for ti in data_mean_centered]) 
	
	mean_shifted=geodesic_riemann(mean_data, move_to,1) 
	data_mean_shifted3 = np.stack([np.dot(sqrtm(mean_shifted), np.dot(ti, sqrtm(mean_shifted))) for ti in data_mean_centered]) 


	# scale dispersion at id
	data_scaled_center = np.stack([powm(ti, 1/2) for ti in data_mean_centered])
	# move back to mean
	data_scaled_to_mean = np.stack([np.dot(sqrtm(mean_data), np.dot(ti, sqrtm(mean_data))) for ti in data_scaled_center]) 

	#for i in np.arange(data_scaled_center.shape[0]):
	#	print(get_volume(data_scaled_center[i])/get_volume(data_mean_centered[i]))


	# rotate
	R=np.array([[1,0,0],
	[0, -0.41614683654,-0.90929742682],
	[0,0.90929742682,-0.41614683654]])

	data_mea_dentered_rotated = np.stack([np.dot(R, np.dot(t, R.T)) for t in data_mean_centered])

	plot_ellipsoids([data_mean_centered, data_mea_dentered_rotated])
	# plot_ellipsoids([data_franka, data_scaled_to_mean])
	# plot_ellipsoids([data_mean_centered, data_scaled_center])
	# plot_ellipsoids([data_franka, data_mean_shifted1,  data_mean_shifted2, data_mean_shifted3, data_mean_centered])


def eigenvalues_analysis():
	data_franka = np.genfromtxt("/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/panda/5000/manipulabilities.csv", delimiter=',')
	data_rhuman = np.genfromtxt("/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/rhuman/5000/manipulabilities.csv", delimiter=',')
	data_fanuc = np.genfromtxt("/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/toy_data/5000/manipulabilities.csv", delimiter=',')
	data_franka = np.reshape(data_franka,(data_franka.shape[0],3,3))
	data_rhuman = np.reshape(data_rhuman,(data_rhuman.shape[0],3,3))
	data_fanuc = np.reshape(data_fanuc,(data_fanuc.shape[0],3,3))

	eigs_franka=np.zeros((data_franka.shape[0],3))
	eigs_rhuman=np.zeros((data_rhuman.shape[0],3))
	eigs_fanuc=np.zeros((data_fanuc.shape[0],3))

	for i in np.arange(data_franka.shape[0]):
		w,v = LA.eig(data_franka[i])
		eigs_franka[i,:] = np.sort(w)
		w2,v2 = LA.eig(data_rhuman[i])
		eigs_rhuman[i,:] = np.sort(w2)
		w3,v3 = LA.eig(data_fanuc[i])
		eigs_fanuc[i,:] = np.sort(w3)

	print("franka: min=%.13f, max=%.13f" %(np.min(eigs_franka), np.max(eigs_franka)))
	print("RHuman: min=%.13f, max=%.13f" %(np.min(eigs_rhuman), np.max(eigs_rhuman)))
	print("Toy data: min=%.13f, max=%.13f" %(np.min(eigs_fanuc), np.max(eigs_fanuc)))

	fig = plt.figure()
	ax = fig.add_subplot(1,3,1,projection='3d')
	ax.scatter(eigs_franka[:,0], eigs_franka[:,1], eigs_franka[:,2], marker='o', s=1)
	ax.set_title("Franka")
	ax2 = fig.add_subplot(1,3,2,projection='3d')
	ax2.scatter(eigs_rhuman[:,0], eigs_rhuman[:,1], eigs_rhuman[:,2], marker='o', s=1)
	ax2.set_title("RHuman")
	ax3 = fig.add_subplot(1,3,3,projection='3d')
	ax3.scatter(eigs_fanuc[:,0], eigs_fanuc[:,1], eigs_fanuc[:,2], marker='o', s=1)
	ax3.set_title("Toy data")
	fig.suptitle("Eigenvalues scatter")

	#plt.show()

	fig = plt.figure()
	ax = fig.add_subplot(1,3,1)
	ax.hist(eigs_franka.flatten(), bins=50, color='cornflowerblue')
	ax.set_title("Franka")
	ax2 = fig.add_subplot(1,3,2)
	ax2.hist(eigs_rhuman.flatten(), bins=50, color='cornflowerblue')
	ax2.set_title("RHuman")
	ax3 = fig.add_subplot(1,3,3)
	ax3.hist(eigs_fanuc.flatten(), bins=50, color='cornflowerblue')
	ax3.set_title("Toy data")
	fig.suptitle("Eigenvalues histogram")

	#plt.show()


	fig = plt.figure()
	ax = fig.add_subplot(1,3,1)

	cond1=np.sqrt(eigs_franka[:,2]/eigs_franka[:,0])
	ax.hist(normalize(cond1), bins=100, color='cornflowerblue')
	ax.set_title("Franka")
	ax2 = fig.add_subplot(1,3,2)
	cond2=np.sqrt(eigs_rhuman[:,2]/eigs_rhuman[:,0])
	ax2.hist(normalize(cond2), bins=100,  color='cornflowerblue')
	ax2.set_title("RHuman")
	ax3 = fig.add_subplot(1,3,3)
	cond3=np.sqrt(eigs_fanuc[:,2]/eigs_fanuc[:,0])
	ax3.hist(normalize(cond3), bins=100,  color='cornflowerblue')
	ax3.set_title("Toy data")
	fig.suptitle("sqrt(lambda_max/lambda_min) (normalized, the bigger the closer to singularity)")


	fig = plt.figure()
	cov1 = np.cov(eigs_franka.transpose())
	cov2 = np.cov(eigs_rhuman.transpose())
	cov3 = np.cov(eigs_fanuc.transpose())

	print("Eigen values of franka cov matrix")
	print(np.log(np.sort(np.linalg.eigvals(cov1))[::-1]))
	print("Eigen values of rhuman cov matrix")
	print(np.log(np.sort(np.linalg.eigvals(cov2))[::-1]))
	print("Eigen values of toy data cov matrix")
	print(np.log(np.sort(np.linalg.eigvals(cov3))[::-1]))

	#cov1 = np.cov(eigs_franka)
	s = np.sort(np.linalg.eigvals(cov1))[::-1]
	ax = fig.add_subplot(1,3,1)
	ax.plot(np.arange(3), s)
	# ax.hist(s, bins=100,  color='cornflowerblue')
	ax.set_title("Franka")
	#cov2 = np.cov(eigs_rhuman)
	s2 = np.sort(np.linalg.eigvals(cov2))[::-1]
	ax2 = fig.add_subplot(1,3,2)
	ax2.plot(np.arange(3), s2)
	# ax2.hist(s2, bins=100,  color='cornflowerblue')
	ax2.set_title("RHuman")
	#cov3 = np.cov(eigs_fanuc)
	s3 = np.sort(np.linalg.eigvals(cov3))[::-1]
	ax3 = fig.add_subplot(1,3,3)
	ax3.plot(np.arange(3), s3)
	# ax3.hist(s3, bins=100,  color='cornflowerblue')
	ax3.set_title("Toy data")
	fig.suptitle("Eigvals of cov ")



	print(np.linalg.eigvals(np.cov(eigs_franka)))
	# w = np.sort(w)
	# ax.hist(w, bins=100, color='cornflowerblue')
	# ax.set_title("Franka")

	# w2,v2 = LA.eig(cov2)
	# w2 = np.sort(w2)
	# ax2 = fig.add_subplot(1,3,2)
	# ax2.hist(w2, bins=100,  color='cornflowerblue')
	# ax2.set_title("RHuman")

	# w3,v3 = LA.eig(cov3)
	# w3 = np.sort(w3)
	# ax3 = fig.add_subplot(1,3,3)
	# ax3.hist(w3, bins=100,  color='cornflowerblue')
	# ax3.set_title("Toy data")
	# fig.suptitle("eigs of the cov matrix")

	plt.show()

if __name__ == "__main__":
	eigenvalues_analysis()