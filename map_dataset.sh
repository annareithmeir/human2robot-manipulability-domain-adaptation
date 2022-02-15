#!/bin/bash
set -e

source py/venv3-6/bin/activate

robot_teacher="panda"
robot_student="toy_data"

base_path="/home/nnrthmr/CLionProjects/ma_thesis/data/mapping"
# map_dataset="reach_up"
map_dataset="10_new"
lookup_dataset="100"

volume_scaling=8
axes_scaling="2,2,1"

redo=0
redo_dataset=0

algorithm=1 # 0: Naive, 1:ICP, 2:CPD-8d, 3:CPD-3d

#echo $redo $redo_dataset

#rm -rf ${base_path}/${robot_student}/${lookup_dataset}/*
if [ $algorithm -eq 0 ]; then
	rm ${base_path}/${robot_student}/${map_dataset}/*mapped*
fi
rm -rf ${base_path}/${robot_student}/${map_dataset}/8d
rm -rf ${base_path}/${robot_teacher}/${map_dataset}/8d
rm -rf ${base_path}/${robot_teacher}/${lookup_dataset}/8d
rm -rf ${base_path}/${robot_student}/${lookup_dataset}/8d

map_data_path="${base_path}/${robot_teacher}/${map_dataset}/manipulabilities_interpolated.csv"
results_path="${base_path}/${robot_student}/${map_dataset}"


echo "###	 Map dataset ${map_dataset} from ${robot_teacher} to ${robot_student} (based on mapping dataset ${lookup_dataset}) 	###"



#generate artificial dataset
if [[ $redo_dataset -eq 1 ]]; then
	rm -rf ${base_path}/${robot_student}/${lookup_dataset}/*
	echo "### 	Create artificial dataset 	###"
	#python3 py/generate_artificial_data.py ${base_path} ${lookup_dataset} ${map_dataset} ${volume_scaling} -axes_scaling ${axes_scaling}
	python3 py/generate_artificial_data_r_s_t.py ${base_path} ${lookup_dataset} ${map_dataset}
	cd /home/nnrthmr/CLionProjects/ma_thesis/matlab
	/usr/local/MATLAB/R2021a/bin/matlab -batch "generateRobotDataToy(${base_path@Q},${lookup_dataset@Q});exit" | tail -n +17
	cd /home/nnrthmr/CLionProjects/ma_thesis
fi



#check if naive lookup table already exists
if [[ $algorithm -eq 0 ]]; then
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

algorithm=1
if [[ $algorithm -eq 1 ]]; then
	echo "###	 Running ICP now	###"
	# python3 -W ignore py/run_icp2.py $base_path $robot_teacher $robot_student $lookup_dataset 2 --map_dataset $map_dataset
	python3 -W ignore py/run_icp.py $base_path $robot_teacher $robot_student $lookup_dataset 2 --map_dataset $map_dataset

	naive_path="${base_path}/${robot_student}/${map_dataset}/manipulabilities_interpolated_mapped_naive.csv"
	icp_path="${base_path}/${robot_student}/${map_dataset}/manipulabilities_interpolated_mapped_icp.csv"
	input_path="${base_path}/${robot_teacher}/${map_dataset}/manipulabilities_interpolated.csv"
	if [ ${robot_student}=="toy_data" ]; then 
		ground_truth_path="${base_path}/${robot_student}/${map_dataset}/manipulabilities_interpolated_groundtruth.csv"
	fi

	echo "###	Plotting 	###"
	paths_list="${input_path},${ground_truth_path},${icp_path}"
	python3 py/plot_artificial_data.py -mapping_paths ${paths_list}
#else
#	echo "### 	ICP mapping parameters found. 	###"
#	echo "### 	Mapping new data with naive ICP	###"
#	python3 -W ignore py/run_icp.py $base_path $robot_teacher $robot_student $lookup_dataset 1 --map_dataset $map_dataset
fi

if [[ $algorithm -eq 2 ]]; then
	echo "###	 Running CPD now	###"
	mkdir ${base_path}/${robot_student}/${lookup_dataset}/8d
	mkdir ${base_path}/${robot_student}/${map_dataset}/8d
	mkdir ${base_path}/${robot_teacher}/${map_dataset}/8d
	mkdir ${base_path}/${robot_teacher}/${lookup_dataset}/8d
	python3 -W ignore py/vectorizeSPD.py SPD_to_8d "$base_path/$robot_teacher/$lookup_dataset/manipulabilities.csv" # vectorize teacher data
	python3 -W ignore py/vectorizeSPD.py SPD_to_8d "$base_path/$robot_student/$lookup_dataset/manipulabilities.csv" # vectorize student data
	python3 -W ignore py/vectorizeSPD.py SPD_to_8d "$base_path/$robot_teacher/$map_dataset/manipulabilities_interpolated.csv" # vectorize teacher map data
	echo "###	 All data vectorized done	###"

	# run cpd and map new points
	cd /home/nnrthmr/CLionProjects/ma_thesis/matlab
	/usr/local/MATLAB/R2021a/bin/matlab -batch "cpd8d(${base_path@Q},${robot_teacher@Q},${robot_student@Q},${lookup_dataset@Q},${map_dataset@Q});exit" | tail -n +17
	cd /home/nnrthmr/CLionProjects/ma_thesis

	echo "###	Plotting 	###"
	cpd_path="${base_path}/${robot_student}/${map_dataset}/8d/manipulabilities_interpolated_mapped_cpd.csv"
	input_path="${base_path}/${robot_teacher}/${map_dataset}/manipulabilities_interpolated.csv"
	ground_truth_path="${base_path}/${robot_student}/${map_dataset}/manipulabilities_interpolated_groundtruth.csv"
	paths_list="${input_path},${ground_truth_path},${cpd_path}"
	python3 py/plot_artificial_data.py -mapping_paths ${paths_list}
fi
echo "Done!"
deactivate
	




