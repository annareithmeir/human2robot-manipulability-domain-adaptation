import numpy as np
import pandas as pd

data_path="../data/"
DATA_PATH='/home/nnrthmr/Desktop/master-thesis/vrep/vrep_franka_promps/py_scripts/data/'
n_demos=4

df=pd.DataFrame()

# first row is zero -> skip
for i in np.arange(n_demos):
	demo_path = data_path + "demos/EEpos_data_trial_"+str(i)+".csv"
	demo = pd.read_csv(demo_path, sep=",")
	demo['s']=demo.index*0.01
	demo=demo.drop(['ns'], axis=1)

	df=df.append(demo.iloc[1:21])

print(df)
df.to_csv("../data/trajectories.csv", index=None)



manipulabilities=list()
df2=pd.DataFrame()
alldata=np.zeros((20*n_demos, 11))
#todo synchronized data, interpolated --> just for the moment cut off at t_min

for i in np.arange(n_demos):
	path=DATA_PATH+'EEpos_translationmanipulability_trial_'+str(i)+'.csv'
	mani = pd.read_csv(path, sep=",")
	mani['0']=mani.index*0.01
	print(mani['0'])
	mani=mani.drop(['1'], axis=1)

	df2=df2.append(mani.iloc[1:21])

df2.to_csv(data_path+"/translationManip3d.csv", index=None)
