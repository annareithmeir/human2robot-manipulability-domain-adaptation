import numpy as np

base_path="/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/panda"


# data1= np.genfromtxt(base_path+"/sing_side_90/manipulabilities_40.csv", delimiter=',')
# data2= np.genfromtxt(base_path+"/sing_side_120/manipulabilities_40.csv", delimiter=',')
# data3= np.genfromtxt(base_path+"/sing_side_150/manipulabilities_40.csv", delimiter=',')
# data4= np.genfromtxt(base_path+"/sing_side_110/manipulabilities_40.csv", delimiter=',')
# #data5= np.genfromtxt(base_path+"/sing_side_140/manipulabilities_40.csv", delimiter=',')
# data6= np.genfromtxt(base_path+"/sing_up_0/manipulabilities_40.csv", delimiter=',')
# data7= np.genfromtxt(base_path+"/sing_up_45/manipulabilities_40.csv", delimiter=',')
# data8= np.genfromtxt(base_path+"/sing_up_90/manipulabilities_40.csv", delimiter=',')
# #data9= np.genfromtxt(base_path+"/sing_up_60/manipulabilities_40.csv", delimiter=',')
# data10= np.genfromtxt(base_path+"/sing_up_30/manipulabilities_40.csv", delimiter=',')

data1= np.genfromtxt(base_path+"/sing_side_90/manipulabilities.csv", delimiter=',')
data2= np.genfromtxt(base_path+"/sing_side_120/manipulabilities.csv", delimiter=',')
data3= np.genfromtxt(base_path+"/sing_side_150/manipulabilities.csv", delimiter=',')
data4= np.genfromtxt(base_path+"/sing_side_110/manipulabilities.csv", delimiter=',')
#data5= np.genfromtxt(base_path+"/sing_side_140/manipulabilities.csv", delimiter=',')
data6= np.genfromtxt(base_path+"/sing_up_0/manipulabilities.csv", delimiter=',')
data7= np.genfromtxt(base_path+"/sing_up_45/manipulabilities.csv", delimiter=',')
data8= np.genfromtxt(base_path+"/sing_up_90/manipulabilities.csv", delimiter=',')
#data9= np.genfromtxt(base_path+"/sing_up_60/manipulabilities.csv", delimiter=',')
data10= np.genfromtxt(base_path+"/sing_up_30/manipulabilities.csv", delimiter=',')

data=np.concatenate((data1, data2, data3, data4, data6, data7, data8, data10))
print(data1.shape)
print(data2.shape)
print(data3.shape)
print(data4.shape)
#print(data5.shape)
print(data6.shape)
print(data7.shape)
print(data8.shape)
#print(data9.shape)
print(data10.shape)
print(data.shape)
np.savetxt(base_path+"/sing_combined/manipulabilities.csv", data, delimiter=',')


