import numpy as np
from dqrobotics import *
from dqrobotics.robot_modeling import DQ_Kinematics
import franka_robot
from scipy.interpolate import interp1d
import pandas as pd


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


COLS=["Franka_joint1","Franka_joint2","Franka_joint3","Franka_joint4","Franka_joint5","Franka_joint6","Franka_joint7"]
COLS2=["s","Franka_joint1","Franka_joint2","Franka_joint3","Franka_joint4","Franka_joint5","Franka_joint6","Franka_joint7"]

BASE_PATH = "/home/nnrthmr/CLionProjects/ma_thesis/data/demos/towards_singularities/panda/"
FILE_PATH = BASE_PATH + "joints.csv"
FILE_PATH_INTERP = BASE_PATH + "joints_interpolated.csv"
FILE_PATH_MANIPS = BASE_PATH + "manipulabilities_interpolated.csv"
FILE_PATH_TS = BASE_PATH + "t_interpolated.csv"


DESIRED_POINTS = 10
DESIRED_SECONDS = 1

# interpolate joints
df = pd.read_csv(FILE_PATH)
df.set_axis(COLS, axis=1, inplace=True)

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
franka = franka_robot.kinematics()
manips = np.zeros((DESIRED_POINTS, 10))
ts= np.zeros((DESIRED_POINTS,4))

for i in np.arange(DESIRED_POINTS):
    q = arr_interp[i,1:]
    Jpose = franka.pose_jacobian(q)
    x_curr = franka.fkm(q)
    J=franka.translation_jacobian(Jpose,x_curr)
    [m, _, _] = franka_robot.get_manipulability_from_translation_jacobian(J)
    manips[i,1:] = m.reshape(1,9)
    ts[i,1:] = vec3(translation(x_curr))

manips[:,0] = x
ts[:,0] = x

df= pd.DataFrame(manips)
df = df.set_index([0])
df.to_csv(FILE_PATH_MANIPS)

df= pd.DataFrame(ts)
df = df.set_index([0])
df.to_csv(FILE_PATH_TS)




