from dqrobotics import *
from dqrobotics.robot_modeling import DQ_SerialManipulatorDH
import math
import numpy as np

def kinematics():

    pi2 = math.pi/2.0;
    franka_dh = np.array([  0,      0,          0,          0,      0,      0,          0,
                        0.333,      0,      0.316,          0,  0.384,      0,      0.107,
                            0,      0,     0.0825,    -0.0825,      0,  0.088,     0.0003,
                         -pi2,    pi2,        pi2,       -pi2,    pi2,    pi2,          0,
                            0,      0,          0,          0,      0,      0,          0]).reshape(5,7)

    franka = DQ_SerialManipulatorDH(franka_dh,"standard")
    return franka

# as in the paper
def get_manipulability(J):
    manip = J.dot(J.T)
    w, v = np.linalg.eig(np.array(manip))
    major_axis = v[:, np.argmax(w)]
    length = np.sqrt(np.max(w))
    return [np.array(manip), major_axis, length]

def get_manipulability_from_translation_jacobian(J):
    J=J[1:,:] # first line zeros
    manip = J.dot(J.T)
    w, v = np.linalg.eig(np.array(manip))
    major_axis = v[:, np.argmax(w)]
    length = np.sqrt(np.max(w))
    return [np.array(manip), major_axis, length]
