#!/bin/bash
set -e

source py/venv3-6/bin/activate

base_path="/home/nnrthmr/CLionProjects/ma_thesis/data/mapping"

# # 2dof toy data
# robot_teacher="2dof"
# robot_student="2dof_vertical"
# # lookup_dataset="500"
# lookup_dataset="links_combined"
# map_datasets="manipulabilities_2 manipulabilities_4 manipulabilities_18"

#rhuman to panda toy data
robot_teacher="rhuman"
robot_student="panda"
#lookup_dataset="10000"
#lookup_dataset="500"
lookup_dataset="sing_combined"
map_datasets="sing_up_60"
#map_datasets="sing_side_140 sing_up_60 side_elbow_fold side_elbow_shoulder_fold"

# robot_teacher="panda"
# robot_student="toy_data"
# lookup_dataset="100"
# map_datasets="10_new"

# real world data
# robot_teacher="rhuman"
# robot_student="panda"
# lookup_dataset="sing_combined"
# #lookup_dataset="10000"
# map_datasets="cut_userchoice_5 cut_optimal_3"

echo "###	 Map dataset ${map_dataset} from ${robot_teacher} to ${robot_student} (based on mapping dataset ${lookup_dataset}) 	###"


redo=0
algorithm=3



if [[ algorithm -eq 0 ]]; then
	#echo "### 	Running createLookupTable now 	###"
	#cd /home/nnrthmr/CLionProjects/ma_thesis/matlab
	#/usr/local/MATLAB/R2021a/bin/matlab -batch "createLookupTable(${base_path@Q},${robot_teacher@Q},${robot_student@Q},${lookup_dataset@Q});exit" | tail -n +17
	#cd /home/nnrthmr/CLionProjects/ma_thesis
	# run naive mapping on new data

	echo "### 	Mapping new data with naive lookup table 	###"
	cd /home/nnrthmr/CLionProjects/ma_thesis/matlab
	for map_dataset in $map_datasets; do
		echo $map_dataset
		/usr/local/MATLAB/R2021a/bin/matlab -batch "map_manipulabilities(${base_path@Q},${robot_teacher@Q},${robot_student@Q},${lookup_dataset@Q},${map_dataset@Q});exit" | tail -n +17
	done
	cd /home/nnrthmr/CLionProjects/ma_thesis
fi

if [[ algorithm -eq 1 ]]; then
	echo "###	 Running ICP now (only find map)	###"
	python3 -W ignore py/run_icp.py $base_path $robot_teacher $robot_student $lookup_dataset 0
fi


algorithm=3
if [[ algorithm -eq 3 ]]; then
	map_path="/home/nnrthmr/CLionProjects/ma_thesis/data/final_results/newestExpHR/PT+ICP/25points/validation"
	# map_path="/home/nnrthmr/CLionProjects/ma_thesis/data/final_results/toydata/validation"
	gt_path="/home/nnrthmr/CLionProjects/ma_thesis/data/final_results/naive"
	for map_dataset in $map_datasets; do
		echo "###	 Running ICP now (only mapping new data: $map_dataset)	###"
		#python3 -W ignore py/run_icp.py $base_path $robot_teacher $robot_student $lookup_dataset 1 --map_dataset $map_dataset
		echo "### 	Mapping new data with ICP	###"
		icp_path="${map_path}/${map_dataset}/manipulabilities_mapped_icp.csv"
		input_path="${base_path}/${robot_teacher}/${map_dataset}/manipulabilities_40.csv" # 2dof
		ground_truth_path="${gt_path}/${map_dataset}/manipulabilities_mapped_naive.csv"
		#ground_truth_path="${base_path}/${robot_student}/${map_dataset}/manipulabilities.csv"
		paths_list="${input_path},${ground_truth_path},${icp_path}"
		echo "###	Plotting 	###"
		python3 py/plot_artificial_data.py -mapping_paths ${paths_list}
	done
fi

# algorithm=3
# if [[ algorithm -eq 3 ]]; then
# 	base_path2="/home/nnrthmr/CLionProjects/ma_thesis/data/final_results/ICP-NNconvpairs/50pointsPerTraj/validation"
# 	for map_dataset in $map_datasets; do
# 		echo "###	 Running ICP now (only mapping new data: $map_dataset)	###"
# 		python3 -W ignore py/run_icp.py $base_path $robot_teacher $robot_student $lookup_dataset 1 --map_dataset $map_dataset
# 		echo "### 	Mapping new data with ICP	###"
# 		icp_path="${base_path2}/${map_dataset}/manipulabilities_mapped_icp.csv"
# 		# icp_path="${base_path2}/validation/${map_dataset}/manipulabilities_interpolated_mapped_icp.csv"
# 		# input_path="${base_path}/${robot_teacher}/${map_dataset}/manipulabilities.csv" # 2dof
# 		# input_path="${base_path}/${robot_teacher}/${map_dataset}/manipulabilities_interpolated.csv" # toy data
# 		input_path="${base_path}/${robot_teacher}/${map_dataset}/manipulabilities_40.csv" # human to robot
# 		# ground_truth_path="${base_path}/${robot_student}/${map_dataset}/manipulabilities.csv"
# 		# ground_truth_path="${base_path}/${robot_student}/${map_dataset}/manipulabilities_interpolated_groundtruth.csv"
# 		ground_truth_path="${base_path}/${robot_student}/${map_dataset}/manipulabilities_40.csv"
# 		# paths_list="${input_path},${icp_path}"
# 		paths_list="${input_path},${ground_truth_path},${icp_path}"
# 		echo "###	Plotting 	###"
# 		python3 py/plot_artificial_data.py -mapping_paths ${paths_list}
# 	done
# fi

# algorithm=3
# if [[ algorithm -eq 3 ]]; then
# 	base_path2="/home/nnrthmr/CLionProjects/ma_thesis/data/final_results/realworld/validation"
# 	for map_dataset in $map_datasets; do
# 		echo "###	 Running ICP now (only mapping new data: $map_dataset)	###"
# 		python3 -W ignore py/run_icp.py $base_path $robot_teacher $robot_student $lookup_dataset 1 --map_dataset $map_dataset
# 		echo "### 	Mapping new data with ICP	###"
# 		icp_path="${base_path2}/${map_dataset}/manipulabilities_mapped_icp.csv"
# 		input_path="${base_path}/${robot_teacher}/${map_dataset}/manipulabilities.csv"
# 		paths_list="${input_path},${icp_path}"
# 		echo "###	Plotting 	###"
# 		python3 py/plot_artificial_data.py -mapping_paths ${paths_list}
# 	done
# fi


if [[ $algorithm -eq 2 ]]; then
	for map_dataset in $map_datasets; do
		echo "###	 Running CPD now	###"
		mkdir -p "/home/nnrthmr/CLionProjects/ma_thesis/data/final_results/2dof-2dofvertical/CPD_8d/validation"
		mkdir -p "/home/nnrthmr/CLionProjects/ma_thesis/data/final_results/2dof-2dofvertical/CPD_8d/validation/$map_dataset"
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
		cpd_path="/home/nnrthmr/CLionProjects/ma_thesis/data/final_results/2dof-2dofvertical/CPD_8d/validation/$map_dataset/manipulabilities_mapped_cpd.csv"
		ground_truth_path="${base_path}/${robot_student}/${map_dataset}/manipulabilities.csv"
		input_path="${base_path}/${robot_teacher}/${map_dataset}/manipulabilities.csv"
		paths_list="${input_path},${ground_truth_path},${cpd_path}"
		python3 py/plot_artificial_data.py -mapping_paths ${paths_list}
	done
fi


echo "Done!"

deactivate
	




