
import numpy as np
from pyriemann.utils.distance import distance_riemann, distance_wasserstein
import argparse



def RMSE(x,y):
	err=0.0
	for i in np.arange(x.shape[0]):
		err+=distance_riemann(x[i],y[i])**2
	return sqrt(err/x.shape[0])



parser = argparse.ArgumentParser()
parser.add_argument("base_path", help="base_path.", type=str)
parser.add_argument("robot_teacher", help="robot_teacher.", type=str)
parser.add_argument("robot_student", help="robot_student", type=str)
parser.add_argument("dataset", help="path to lookup dataset e.g. 5000", type=str)
parser.add_argument("--cv_k", help="fold number", type=str, nargs='?')
args = parser.parse_args()

# evaluate current fold of crossvalidation
if args.cv_k is not None:
	err_n = 0.0
	err_icp = 0.0
	args.cv_k = int(args.cv_k)
	mapped_naive=np.genfromtxt(args.base_path+"/"+args.robot_student+"/"+args.dataset+"/cv/manipulabilities_mapped_naive.csv", delimiter=",")
	mapped_naive=mapped_naive.reshape((mapped_naive.shape[0],3,3))
	print(mapped_naive)

	mapped_icp=np.genfromtxt(args.base_path+"/"+args.robot_student+"/"+args.dataset+"/cv/manipulabilities_mapped_icp.csv", delimiter=",")
	mapped_icp=mapped_icp.reshape((mapped_icp.shape[0],3,3))

	manips_g=np.genfromtxt(args.base_path+"/"+args.robot_student+"/"+args.dataset+"/manipulabilities.csv", delimiter=",")
	manips_g=manips_g.reshape((manips_g.shape[0],3,3))
	idx = np.genfromtxt(args.base_path+"/"+args.robot_student+"/"+args.dataset+"/cv/cv_idx.csv", delimiter=',')
	manips_g=manips_g[np.where(idx[args.cv_k,:]==1)]

	assert(mapped_icp.shape[0] == mapped_naive.shape[0]== manips_g.shape[0])

	print("Errors per sample in new dataset")
	for i in np.arange(mapped_icp.shape[0]):
	    m_g=manips_g[i]
	    m_icp=mapped_icp[i]
	    m_naive=mapped_naive[i]
	    err_n+=distance_riemann(m_g, m_naive)
	    err_icp+=distance_riemann(m_g, m_icp)
	    print("%.3f, %.3f" %(distance_riemann(m_g, m_naive), distance_riemann(m_g, m_icp)))
	err_n=float(err_n)/float(mapped_icp.shape[0])
	err_icp=float(err_icp)/float(mapped_icp.shape[0])


	with open(args.base_path+"/"+args.robot_student+"/"+args.dataset+"/cv/errs_naive_icp.txt", "a") as file_object:
	    file_object.write(str(err_n)+"	"+str(err_icp)+"\n")

# evaluate all folds after crossvalidation
else:
	errs=np.genfromtxt(args.base_path+"/"+args.robot_student+"/"+args.dataset+"/cv/errs_naive_icp.txt", delimiter="	")


	with open(args.base_path+"/"+args.robot_student+"/"+args.dataset+"/cv/errs_naive_icp.txt", "a") as file_object:
	    file_object.write("-----------------------------\n")
	    file_object.write(str(np.mean(errs[:,0]))+"	"+str(np.mean(errs[:,1]))+"	(mean)"+"\n")
	    file_object.write(str(np.std(errs[:,0]))+"	"+str(np.std(errs[:,1]))+"	(std)"+"\n")


