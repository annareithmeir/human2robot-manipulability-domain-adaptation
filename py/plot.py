import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pandas as pd

colors=['green', 'blue', 'orange', 'red', 'purple']

fig = plt.figure()
ax = plt.axes(projection='3d')
plt.title('Demonstrations')

'''
for i in np.arange(5):
	data_path = "../data/demos/EEpos_data_trial_"+str(i)+".csv"
	data = pd.read_csv(data_path, sep=",")

	xdata= np.array(data['EE_x'])
	ydata= np.array(data['EE_y'])
	zdata= np.array(data['EE_z'])

	ax.scatter3D(xdata, ydata, zdata, c=colors[i])
'''

data_path = "/home/nnrthmr/C_Mat.csv"
data = pd.read_csv(data_path, sep=",")
data=data.append(data.loc[1]*0.75)

xdata= np.array(data.iloc[0])
ydata= np.array(data.iloc[1])
zdata= np.array(data.iloc[2])

ax.scatter3D(xdata, ydata, zdata, c=colors[0])

plt.show()
