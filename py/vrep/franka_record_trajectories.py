import sim as vrep 
import csv
import sys
import math
import time
import pandas as pd
from dqrobotics import *
from dqrobotics.interfaces.vrep import DQ_VrepInterface
import franka_robot
import numpy as np

JOINTS = [ 'Franka_joint1', 'Franka_joint2', 'Franka_joint3', 'Franka_joint4',
    'Franka_joint5', 'Franka_joint6', 'Franka_joint7'
    ]
Q_MAX=[2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
Q_MIN=[-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]

## horizontal paths


vi = DQ_VrepInterface()

try:
    # Connects to the localhost in port 19997 with timeout 100ms and 10 retries for each method call
    if not vi.connect(19999, 100, 5):
        raise Exception("Unable to connect to vrep!")

    # Starts simulation in V-REP
    print("Starting V-REP simulation...")
    vi.start_simulation()

    franka = franka_robot.kinematics() 

    # horizontal path: 
    q = [0, 0, 0.0, 0, 0.0, 0, 0.0]

    for q1 in np.arange():
        franka_q = vi.set_joint_positions(JOINTS,q)
        sleep(20)


    vi.stop_simulation()
    vi.disconnect()

except Exception as exp:
    print(exp)
    vi.stop_simulation()
    vi.disconnect_all()



# null space projectors


