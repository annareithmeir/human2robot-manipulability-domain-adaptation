#!/bin/bash
set -e

source py/venv3-6/bin/activate

robot_teacher="2dof"
robot_student="2dof_scaled"
base_path="/home/nnrthmr/CLionProjects/ma_thesis/data/mapping"
lookup_dataset="links_combined"
map_datasets="manipulabilities_2 manipulabilities_4"

#robot_teacher="rhuman"
#robot_student="panda"
#lookup_dataset="sing_combined"
#map_datasets="sing_side_140 sing_up_60 side_elbow_fold side_elbow_shoulder_fold"


#map_data_path="${base_path}/${robot_teacher}/${map_dataset}/manipulabilities.csv"
#results_path="${base_path}/${robot_student}/${map_dataset}"


echo "###	 Map dataset ${map_dataset} from ${robot_teacher} to ${robot_student} (based on mapping dataset ${lookup_dataset}) 	###"


redo=0
algorithm=1 # 0: Naive, 1:ICP, 2:CPD-8d, 3:ICP-validation-only



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
	echo "###	 Running ICP now (only find map)	###"
	python3 -W ignore py/run_icp.py $base_path $robot_teacher $robot_student $lookup_dataset 0
fi

algorithm=3
if [[ algorithm -eq 3 ]]; then
	for map_dataset in $map_datasets; do
		echo "###	 Running ICP now (only mapping new data: $map_dataset)	###"
		python3 -W ignore py/run_icp.py $base_path $robot_teacher $robot_student $lookup_dataset 1 --map_dataset $map_dataset
		echo "### 	Mapping new data with ICP	###"
		icp_path="${base_path}/${robot_student}/${map_dataset}/manipulabilities_mapped_icp.csv"
		input_path="${base_path}/${robot_teacher}/${map_dataset}/manipulabilities.csv"
		ground_truth_path="${base_path}/${robot_student}/${map_dataset}/manipulabilities.csv"
		paths_list="${input_path},${ground_truth_path},${icp_path}"
		echo "###	Plotting 	###"
		python3 py/plot_artificial_data.py -mapping_paths ${paths_list}
	done
fi


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

# echo "###	Plotting 	###"
# cpd_path="${base_path}/${robot_student}/${map_dataset}/8d/manipulabilities_mapped_cpd.csv"
# input_path="${base_path}/${robot_teacher}/${map_dataset}/manipulabilities.csv"
# icp_path="${base_path}/${robot_student}/${map_dataset}/manipulabilities_mapped_icp.csv"
# paths_list="${input_path},${icp_path},${cpd_path}"
# python3 py/plot_artificial_data.py -mapping_paths ${paths_list}


deactivate
	




