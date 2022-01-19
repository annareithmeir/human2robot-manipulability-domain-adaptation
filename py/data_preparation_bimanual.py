import numpy as np
from dqrobotics import *
from dqrobotics.robot_modeling import DQ_Kinematics, DQ_CooperativeDualTaskSpace
import franka_robot
from scipy.interpolate import interp1d
import pandas as pd

def geomJ(J, q):

    C8 = np.diag([-1, 1, 1, 1,-1, 1, 1, 1])
    C4m= -C8[:4,:4]
    CJ4_2_J3=np.zeros((3,4))
    CJ4_2_J3[0,1]=1
    CJ4_2_J3[1,2]=1
    CJ4_2_J3[2,3]=1

    J1=CJ4_2_J3 @ (2*haminus4(q.P().conj())) @ (J[:4,:])
    J2=CJ4_2_J3 @ (2*(hamiplus4(q.D()) @ (C4m) @ (J[:4,:]) + haminus4(q.P().conj()) @ (J[4:,:])))

    return np.concatenate((J1,J2), axis=0)


def interpolate_equidistant(x, trajectory, interpolate_N_points):
    # # trajectory as np.array (N_deg_of_freedom, T)
    # # interpolate_N_points as int, indicating the number of timesteps of resulting aligned trajectories
    # assert trajectory.shape[1] > 0

    # # calculate the distances between each consecutive element
    # all_diffs = np.diff(trajectory, axis=1)
    # dist = np.sqrt(np.sum(all_diffs ** 2, axis=0))  # euclidean distance, no division required
    # # and also make as long as the original trajectory again
    # # take that as the x value of the interpolation
    # u = np.hstack(([0], np.cumsum(dist)))

    # # filter out repeatedly occurring elements
    # some_change_in_space = np.hstack(([True], dist >= 1e-16))
    # u = np.compress(some_change_in_space, u)
    # trajectory = np.compress(some_change_in_space, trajectory, axis=1)

    # sample at the desired number of points
    t = np.linspace(0, x.max(initial=-np.inf), interpolate_N_points)

    return interp1d(x, trajectory, assume_sorted=True, kind='cubic')(t)


#q=DQ(0.0032449 ,0.6685, - 0.0029168, 0.7437, 0.038786, 0.0019166, 0.43927, - 0.00016923)
#J=np.genfromtxt("/home/nnrthmr/tmp.csv", delimiter=',')
#J=J.reshape((8,14))


COLS=["Franka1_joint1","Franka1_joint2","Franka1_joint3","Franka1_joint4","Franka1_joint5","Franka1_joint6","Franka1_joint7",
        "Franka2_joint1","Franka2_joint2","Franka2_joint3","Franka2_joint4","Franka2_joint5","Franka2_joint6","Franka2_joint7"]
COLS2=["s","Franka1_joint1","Franka1_joint2","Franka1_joint3","Franka1_joint4","Franka1_joint5","Franka1_joint6","Franka1_joint7",
        "Franka2_joint1","Franka2_joint2","Franka2_joint3","Franka2_joint4","Franka2_joint5","Franka2_joint6","Franka2_joint7"]

BASE_PATH = "/home/nnrthmr/CLionProjects/ma_thesis/data/demos/bimanual/reach_up/"
FILE_PATH = BASE_PATH + "joints.csv"
FILE_PATH_INTERP = BASE_PATH + "joints_interpolated.csv"
FILE_PATH_MANIPS = BASE_PATH + "manipulabilities_interpolated.csv"
FILE_PATH_TS = BASE_PATH + "t_interpolated.csv"


DESIRED_POINTS = 10
DESIRED_SECONDS = 1

# interpolate joints
df = pd.read_csv(FILE_PATH)
df.set_axis(COLS2, axis=1, inplace=True)

arr = df[COLS].to_numpy()

xorig = np.linspace(0, DESIRED_SECONDS, arr.shape[0])
x = np.linspace(0, DESIRED_SECONDS, DESIRED_POINTS)

arr_interp = interpolate_equidistant(xorig, arr.T, DESIRED_POINTS).T  # promp expects (N_steps, D) trajectories
arr_interp = np.insert(arr_interp, 0, 0, axis=1)
arr_interp[:,0] = x

df= pd.DataFrame(arr_interp)
df.set_axis(COLS2, axis=1, inplace=True)
df = df.set_index('s')

df.to_csv(FILE_PATH_INTERP)

# generate manipulabilities interpolated
franka1 = franka_robot.kinematics()
franka2 = franka_robot.kinematics()

panda_bimanual = DQ_CooperativeDualTaskSpace(franka1, franka2)


manips = np.zeros((DESIRED_POINTS, 10))
ts= np.zeros((DESIRED_POINTS,4))

for i in np.arange(DESIRED_POINTS):
    q = arr_interp[i,1:]
    pose_abs = panda_bimanual.absolute_pose(q); 
    pose_abs_j = panda_bimanual.absolute_pose_jacobian(q); # 8x14 matrix
    j_geom = geomJ(pose_abs_j, pose_abs); # 6x14

    j = j_geom[:3,:]
    m=j @ j.transpose()

    x_curr = vec3(translation(pose_abs))

    #q = arr_interp[i,1:]
    #Jpose = franka.pose_jacobian(q)
    #x_curr = franka.fkm(q)
    #J=franka.translation_jacobian(Jpose,x_curr)
    #[m, _, _] = franka_robot.get_manipulability_from_translation_jacobian(J)

    manips[i,1:] = m.reshape(1,9)
    ts[i,1:] = x_curr

manips[:,0] = x
ts[:,0] = x

df= pd.DataFrame(manips)
df = df.set_index([0])
df.to_csv(FILE_PATH_MANIPS)

df= pd.DataFrame(ts)
df = df.set_index([0])
df.to_csv(FILE_PATH_TS)




