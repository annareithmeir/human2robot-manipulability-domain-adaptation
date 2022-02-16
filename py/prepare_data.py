import numpy as np
import pandas as pd 
import sys
from pathlib import Path
import glob


agg_all = False
agg_proband = 5

task='cut_userchoice'
data_path="/home/nnrthmr/CLionProjects/ma_thesis/data/demos/rhuman_luis/data/"+task
# data_path="/home/nnrthmr/CLionProjects/ma_thesis/data/demos/rhuman_luis/data/"+task+"/interpolated"
#data_path="data/"+task

demo=pd.DataFrame(columns=['EE_x','EE_y','EE_z'])
demo_m=pd.DataFrame(columns=['0','1','2','3','4','5','6','7','8','9'])
lens=list()

if not agg_all:
	files=sorted(glob.glob(data_path+"/exp"+str(agg_proband)+"*_t.csv"))
	files_m=sorted(glob.glob(data_path+"/exp"+str(agg_proband)+"*_m.csv"))
else:
	files=sorted(glob.glob(data_path+"/*_t.csv"))
	files_m=sorted(glob.glob(data_path+"/*_m.csv"))

assert(len(files)==len(files_m))


for file in files:
	demo_tmp = pd.read_csv(file, sep=",", names=['EE_x','EE_y','EE_z'])
	lens.append(len(demo_tmp.index))

for file in files:
	demo_tmp = pd.read_csv(file, sep=",", names=['EE_x','EE_y','EE_z'])
	demo_tmp=demo_tmp.iloc[:min(lens)]
	demo= demo.append(demo_tmp)


for file in files_m:
	demo_tmp = pd.read_csv(file, sep=",", names=['0','1','2','3','4','5','6','7','8','9'])
	demo_tmp=demo_tmp.iloc[:min(lens)]
	demo_m= demo_m.append(demo_tmp)

if not agg_all:
	Path(data_path+"/agg/"+str(agg_proband)).mkdir(parents=True, exist_ok=True)
	demo.to_csv(data_path+"/agg/"+str(agg_proband)+"/all_t.csv")
	demo_m.to_csv(data_path+"/agg/"+str(agg_proband)+"/all_m.csv", index=False)

	with open(data_path+"/agg/"+str(agg_proband)+"/readme.txt", "w") as text_file:
		text_file.write(' Number of points per demo: %i\n Number of demos: %i\n Total number of points: %i' % (min(lens), demo.shape[0]/min(lens), demo.shape[0]))

	with open(data_path+"/agg/"+str(agg_proband)+"/info.txt", "w") as text_file:
		text_file.write('%i %i %i' % (min(lens), demo.shape[0]/min(lens), demo.shape[0]))

else:
	Path(data_path+"/agg").mkdir(parents=True, exist_ok=True)
	demo.to_csv(data_path+"/agg/all_t.csv")
	demo_m.to_csv(data_path+"/agg/all_m.csv", index=False)

	with open(data_path+"/agg/readme.txt", "w") as text_file:
		text_file.write(' Number of points per demo: %i\n Number of demos: %i\n Total number of points: %i' % (min(lens), demo.shape[0]/min(lens), demo.shape[0]))

	with open(data_path+"/agg/info.txt", "w") as text_file:
		text_file.write('%i %i %i' % (min(lens), demo.shape[0]/min(lens), demo.shape[0]))

print(lens)
print('Number of points per demo:')
print(min(lens))
print('Number of demos: ')
print(demo.shape[0]/min(lens))
print('Total number of points: ')
print(demo.shape[0])






