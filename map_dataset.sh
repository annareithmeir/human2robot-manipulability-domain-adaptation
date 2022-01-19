#!/bin/bash
set -e

robot_teacher="panda"
robot_student="toy_data"
base_path="/home/nnrthmr/CLionProjects/ma_thesis/data/mapping"
map_dataset="10_new"
lookup_dataset="100"
volume_scaling=4
axes_scaling="5,1,0.2"

redo=1

echo "###	 Map dataset ${map_dataset} from ${robot_teacher} to ${robot_student} (based on mapping dataset ${lookup_dataset}) 	###"
source py/venv3-6/bin/activate


map_data_path="${base_path}/${robot_teacher}/${map_dataset}/manipulabilities_interpolated.csv"
results_path="${base_path}/${robot_student}/${map_dataset}"


#generate artificial dataset
if [ ! -f "${base_path}/${robot_student}/${lookup_dataset}/manipulabilities.csv" ] || [ redo==1 ]; then
#if [ ! -f "${base_path}/${robot_student}/${lookup_dataset}/manipulabilities.csv" ]; then
	echo "### 	Artificial dataset not found. Creating now ... 	###"
	echo "### 	manipulabilities.csv 	###"
	python3 py/generate_artificial_data.py ${base_path} ${lookup_dataset} ${map_dataset} ${volume_scaling} -axes_scaling ${axes_scaling}
fi
#if [ ! -f "${base_path}/${robot_student}/${lookup_dataset}/scales.csv" ]; then
if [ ! -f "${base_path}/${robot_student}/${lookup_dataset}/scales.csv" ] || [ redo==1 ]; then
	echo "### 	manipulabilities_normalized.csv 	###"
	echo "### 	scales.csv 	###"
	cd /home/nnrthmr/CLionProjects/ma_thesis/matlab
	/usr/local/MATLAB/R2021a/bin/matlab -batch "generateRobotDataToy(${base_path@Q},${lookup_dataset@Q});exit" | tail -n +17
	cd /home/nnrthmr/CLionProjects/ma_thesis
else
	echo "###	 Artificial dataset found. 	###"
fi

#check if naive lookup table already exists
if [ ! -f "${base_path}/${robot_teacher}/${lookup_dataset}/lookup_trafos_naive_${robot_teacher}_to_${robot_student}.csv" ] || [ redo==1 ]; then
#	if [ ! -f "${base_path}/${robot_teacher}/${lookup_dataset}/lookup_trafos_naive_${robot_teacher}_to_${robot_student}.csv" ] ; then
	echo "### 	Naive lookup table not found. Running createLookupTable now... 	###"
	cd /home/nnrthmr/CLionProjects/ma_thesis/matlab
	/usr/local/MATLAB/R2021a/bin/matlab -batch "createLookupTable(${base_path@Q},${robot_teacher@Q},${robot_student@Q},${lookup_dataset@Q});exit" | tail -n +17
	cd /home/nnrthmr/CLionProjects/ma_thesis
else
	echo "###	 Naive lookup table found. 	###"
fi

# run naive mapping on new data
echo "### 	Mapping new data with naive lookup table... 	###"
cd /home/nnrthmr/CLionProjects/ma_thesis/matlab
/usr/local/MATLAB/R2021a/bin/matlab -batch "map_manipulabilities(${base_path@Q},${robot_teacher@Q},${robot_student@Q},${lookup_dataset@Q},${map_dataset@Q});exit" | tail -n +17
cd /home/nnrthmr/CLionProjects/ma_thesis
echo "Done!"

#check if icp parameters already exist and run icp on new data
echo $base_path $robot_teacher $robot_student $lookup_dataset $map_dataset 2


if [ ! -f "${base_path}/${robot_teacher}/${lookup_dataset}/R_icp_${robot_teacher}_to_${robot_student}.txt" ] || [ redo==1 ]; then
	echo "###	 ICP mapping parameters not found. Running ICP now... 	###"
	python3 -W ignore py/run_icp.py $base_path $robot_teacher $robot_student $lookup_dataset $map_dataset 2
else
	echo "### 	ICP mapping parameters found. 	###"
	echo "### 	Mapping new data with naive ICP... 	###"
	python3 -W ignore py/run_icp.py $base_path $robot_teacher $robot_student $lookup_dataset $map_dataset 1
fi
echo "Done!"

naive_path="${base_path}/${robot_student}/${map_dataset}/manipulabilities_interpolated_mapped_naive.csv"
icp_path="${base_path}/${robot_student}/${map_dataset}/manipulabilities_interpolated_mapped_icp.csv"
input_path="${base_path}/${robot_teacher}/${map_dataset}/manipulabilities_interpolated.csv"
if [ ${robot_student}=="toy_data" ]; then 
	ground_truth_path="${base_path}/${robot_student}/${map_dataset}/manipulabilities_interpolated_groundtruth.csv"
fi

echo "###	Plotting... 	###"
python3 py/plot_artificial_data.py ${input_path} ${naive_path} ${icp_path} ${ground_truth_path}
echo "### 	Done! 	###"
deactivate
	




