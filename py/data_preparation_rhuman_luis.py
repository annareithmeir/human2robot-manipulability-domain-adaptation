import numpy as np
from dqrobotics import *
from dqrobotics.robot_modeling import DQ_Kinematics
import franka_robot
from scipy.interpolate import interp1d
import pandas as pd
import os


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



COLS=["joint1","joint2","joint3","joint4","joint5","joint6","joint7"]
COLS2=["s","joint1","joint2","joint3","joint4","joint5","joint6","joint7"]

BASE_PATH = "/home/nnrthmr/CLionProjects/ma_thesis/data/demos/rhuman_luis/data/cut_random/"
OUT_PATH = BASE_PATH+"interpolated/"

if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)


for filename in os.listdir(BASE_PATH):
    if filename.endswith("_joints.csv") :
        print(filename)
        FILE_PATH = BASE_PATH+filename

        # interpolate joints
        df = pd.read_csv(FILE_PATH, header=None)
        df.set_axis(COLS2, axis=1, inplace=True)
        arr = df[COLS].to_numpy()
        x=df['s'].to_numpy()

        DESIRED_POINTS = 100
        DESIRED_SECONDS = x.max()

        arr_interp = interpolate_equidistant(x, arr.T, DESIRED_POINTS).T  # promp expects (N_steps, D) trajectories
        #arr_interp=np.reshape(arr_interp, (-1, 1))
        #print(arr_interp)
        arr_interp = np.insert(arr_interp, 0, 0, axis=1)
        arr_interp[:,0] = np.linspace(0,DESIRED_SECONDS, DESIRED_POINTS)

        df= pd.DataFrame(arr_interp)
        df.set_axis(COLS2, axis=1, inplace=True)
        df = df.set_index('s')

        df.to_csv(OUT_PATH+filename)

        # # generate manipulabilities interpolated
        # franka = franka_robot.kinematics()
        # manips = np.zeros((DESIRED_POINTS, 9))
        # for i in np.arange(DESIRED_POINTS):
        #     q = arr_interp[i,1:]
        #     Jpose = franka.pose_jacobian(q)
        #     x_curr = franka.fkm(q)
        #     J=franka.translation_jacobian(Jpose,x_curr)
        #     [m, _, _] = franka_robot.get_manipulability_from_translation_jacobian(J)
        #     manips[i,:] = m.reshape(1,9)

        # manips = np.insert(manips, 0, 0, axis=1)
        # manips[:,0] = np.linspace(0,DESIRED_SECONDS, DESIRED_POINTS)
        # df= pd.DataFrame(manips)
        # df = df.set_index([0])
        # df.to_csv(OUT_PATH+filename)




