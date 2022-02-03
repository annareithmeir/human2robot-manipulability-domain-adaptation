#!/bin/bash
set -e

source py/venv3-6/bin/activate

robot_teacher="rhuman"
robot_student="panda"
base_path="/home/nnrthmr/CLionProjects/ma_thesis/data/mapping"
map_dataset="10"
lookup_dataset="100"


map_data_path="${base_path}/${robot_teacher}/${map_dataset}/manipulabilities_interpolated.csv"
results_path="${base_path}/${robot_student}/${map_dataset}"


echo "###	 Map dataset ${map_dataset} from ${robot_teacher} to ${robot_student} (based on mapping dataset ${lookup_dataset}) 	###"


redo=1


#check if naive lookup table already exists
if [ redo==1 ]; then
	echo "### 	Running createLookupTable now 	###"
	cd /home/nnrthmr/CLionProjects/ma_thesis/matlab
	/usr/local/MATLAB/R2021a/bin/matlab -batch "createLookupTable(${base_path@Q},${robot_teacher@Q},${robot_student@Q},${lookup_dataset@Q});exit" | tail -n +17
	cd /home/nnrthmr/CLionProjects/ma_thesis
else
	echo "###	 Naive lookup table found. 	###"
fi

if [ redo==1 ]; then
# run naive mapping on new data
echo "### 	Mapping new data with naive lookup table 	###"
cd /home/nnrthmr/CLionProjects/ma_thesis/matlab
/usr/local/MATLAB/R2021a/bin/matlab -batch "map_manipulabilities(${base_path@Q},${robot_teacher@Q},${robot_student@Q},${lookup_dataset@Q},${map_dataset@Q});exit" | tail -n +17
cd /home/nnrthmr/CLionProjects/ma_thesis
echo "Done!"
fi

#check if icp parameters already exist and run icp on new data

if [ redo>=1 ]; then
	echo "###	 Running ICP now	###"
	python3 -W ignore py/run_icp.py $base_path $robot_teacher $robot_student $lookup_dataset 2 --map_dataset $map_dataset
else
	echo "### 	ICP mapping parameters found. 	###"
	echo "### 	Mapping new data with naive ICP	###"
	python3 -W ignore py/run_icp.py $base_path $robot_teacher $robot_student $lookup_dataset 1 --map_dataset $map_dataset
fi
echo "Done!"

naive_path="${base_path}/${robot_student}/${map_dataset}/manipulabilities_interpolated_mapped_naive.csv"
icp_path="${base_path}/${robot_student}/${map_dataset}/manipulabilities_interpolated_mapped_icp.csv"
input_path="${base_path}/${robot_teacher}/${map_dataset}/manipulabilities_interpolated_normalized.csv"
if [ ${robot_student}=="toy_data" ]; then 
	ground_truth_path="${base_path}/${robot_student}/${map_dataset}/manipulabilities_interpolated_groundtruth_normalized.csv"
fi

echo "###	Plotting 	###"
python3 py/plot_artificial_data.py ${input_path} ${naive_path} ${icp_path} ${ground_truth_path}
echo "### 	Done! 	###"
deactivate
	




