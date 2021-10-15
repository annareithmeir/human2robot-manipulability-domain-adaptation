import numpy as np
import pandas as pd
from dqrobotics import *
from dqrobotics.robot_modeling import DQ_Kinematics
import franka_robot

joints=["panda_right_joint1",	"panda_right_joint2",	"panda_right_joint3",	"panda_right_joint4",	"panda_right_joint5",	"panda_right_joint6",	"panda_right_joint7"]
data_path="/home/nnrthmr/Desktop/master-thesis/promps-code/tum_tuda_project/recorded_data/Session_25_06_2021/"

columns=['0', '1', '2', '3', '4', '5' ,'6', '7','8','9']
data=np.zeros((400,10))
franka = franka_robot.kinematics()

for i in np.arange(1,5):
	path=data_path+'run'+str(i)+'.csv'
	m=pd.read_csv(path)
	m=m.iloc[1:101]
	m=np.array(m[joints])
	
	for t in np.arange(100):
		q = m[t,:]
		Jpose = franka.pose_jacobian(q)
		x_curr = franka.fkm(q)
		J=franka.translation_jacobian(Jpose,x_curr)
		[mani, _, _] = franka_robot.get_manipulability_from_translation_jacobian(J)
		tmp=mani.ravel()
		data[(i-1)*100+t,0]= 0.01*t + 0.01
		data[(i-1)*100+t,1:]=tmp
	
df = pd.DataFrame(data, index=None, columns=columns)
print(df)
df.to_csv("../data/demos/tum_tuda/translationManip3d.csv", index=None)
