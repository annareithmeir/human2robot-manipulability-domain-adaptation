#!/bin/bash
set -e

source py/venv3-6/bin/activate

robot_teacher="rhuman"
robot_student="panda"
base_path="/home/nnrthmr/CLionProjects/ma_thesis/data/mapping"
map_dataset="sing_side_140"
# map_dataset="sing_up_60"
map_dataset="side_elbow_fold"
lookup_dataset="sing_combined"


map_data_path="${base_path}/${robot_teacher}/${map_dataset}/manipulabilities.csv"
results_path="${base_path}/${robot_student}/${map_dataset}"


echo "###	 Map dataset ${map_dataset} from ${robot_teacher} to ${robot_student} (based on mapping dataset ${lookup_dataset}) 	###"


redo=0
algorithm=1 # 0: Naive, 1:ICP, 2:CPD-8d, 3:CPD-3d



if [[ algorithm -eq 0 ]]; then
	echo "### 	Running createLookupTable now 	###"
	cd /home/nnrthmr/CLionProjects/ma_thesis/matlab
	/usr/local/MATLAB/R2021a/bin/matlab -batch "createLookupTable(${base_path@Q},${robot_teacher@Q},${robot_student@Q},${lookup_dataset@Q});exit" | tail -n +17
	cd /home/nnrthmr/CLionProjects/ma_thesis
	# run naive mapping on new data
	echo "### 	Mapping new data with naive lookup table 	###"
	cd /home/nnrthmr/CLionProjects/ma_thesis/matlab
	/usr/local/MATLAB/R2021a/bin/matlab -batch "map_manipulabilities(${base_path@Q},${robot_teacher@Q},${robot_student@Q},${lookup_dataset@Q},${map_dataset@Q});exit" | tail -n +17
	cd /home/nnrthmr/CLionProjects/ma_thesis
fi

if [[ algorithm -eq 1 ]]; then
	echo "###	 Running ICP now	###"
	python3 -W ignore py/run_icp.py $base_path $robot_teacher $robot_student $lookup_dataset 2 --map_dataset $map_dataset
	echo "### 	Mapping new data with ICP	###"
	icp_path="${base_path}/${robot_student}/${map_dataset}/manipulabilities_mapped_icp.csv"
	input_path="${base_path}/${robot_teacher}/${map_dataset}/manipulabilities_40.csv"
	#ground_truth_path="${base_path}/${robot_student}/${map_dataset}/manipulabilities.csv"
	# paths_list="${input_path},${ground_truth_path},${icp_path}"
	paths_list="${input_path},${icp_path}"
	echo "###	Plotting 	###"
	python3 py/plot_artificial_data.py -mapping_paths ${paths_list}
fi

if [[ algorithm -eq 3 ]]; then
	echo "###	 Running ICP now	###"
	python3 -W ignore py/run_icp.py $base_path $robot_teacher $robot_student $lookup_dataset 1 --map_dataset $map_dataset
	echo "### 	Mapping new data with ICP	###"
	icp_path="${base_path}/${robot_student}/${map_dataset}/manipulabilities_mapped_icp.csv"
	input_path="${base_path}/${robot_teacher}/${map_dataset}/manipulabilities_40.csv"
	#ground_truth_path="${base_path}/${robot_student}/${map_dataset}/manipulabilities.csv"
	# paths_list="${input_path},${ground_truth_path},${icp_path}"
	paths_list="${input_path},${icp_path}"
	echo "###	Plotting 	###"
	python3 py/plot_artificial_data.py -mapping_paths ${paths_list}
fi


if [[ $algorithm -eq 2 ]]; then
	echo "###	 Running CPD now	###"
	mkdir -p ${base_path}/${robot_student}/${lookup_dataset}/8d
	mkdir -p ${base_path}/${robot_student}/${map_dataset}/8d
	mkdir -p ${base_path}/${robot_teacher}/${map_dataset}/8d
	mkdir -p ${base_path}/${robot_teacher}/${lookup_dataset}/8d
	python3 -W ignore py/vectorizeSPD.py SPD_to_8d "$base_path/$robot_teacher/$lookup_dataset/manipulabilities.csv" # vectorize teacher data
	python3 -W ignore py/vectorizeSPD.py SPD_to_8d "$base_path/$robot_student/$lookup_dataset/manipulabilities.csv" # vectorize student data
	python3 -W ignore py/vectorizeSPD.py SPD_to_8d "$base_path/$robot_teacher/$map_dataset/manipulabilities.csv" # vectorize teacher map data
	echo "###	 All data vectorized done	###"

	# run cpd and map new points
	cd /home/nnrthmr/CLionProjects/ma_thesis/matlab
	/usr/local/MATLAB/R2021a/bin/matlab -batch "cpd8d(${base_path@Q},${robot_teacher@Q},${robot_student@Q},${lookup_dataset@Q},${map_dataset@Q});exit" | tail -n +17
	cd /home/nnrthmr/CLionProjects/ma_thesis

	echo "###	Plotting 	###"
	cpd_path="${base_path}/${robot_student}/${map_dataset}/8d/manipulabilities_mapped_cpd.csv"
	ground_truth_path="${base_path}/${robot_student}/${map_dataset}/manipulabilities.csv"
	input_path="${base_path}/${robot_teacher}/${map_dataset}/manipulabilities.csv"
	paths_list="${input_path},${ground_truth_path},${cpd_path}"
	python3 py/plot_artificial_data.py -mapping_paths ${paths_list}
fi


echo "Done!"

# echo "###	Plotting 	###"
# cpd_path="${base_path}/${robot_student}/${map_dataset}/8d/manipulabilities_mapped_cpd.csv"
# input_path="${base_path}/${robot_teacher}/${map_dataset}/manipulabilities.csv"
# icp_path="${base_path}/${robot_student}/${map_dataset}/manipulabilities_mapped_icp.csv"
# paths_list="${input_path},${icp_path},${cpd_path}"
# python3 py/plot_artificial_data.py -mapping_paths ${paths_list}


deactivate
	




