#!/bin/bash
set -e

source py/venv3-6/bin/activate

robot_teacher="panda"
robot_student="toy_data"
base_path="/home/nnrthmr/CLionProjects/ma_thesis/data/mapping"
dataset="100"

k="5"
k_int=5

generate_data=1
volume_scaling=4
axes_scaling="1,1,1"


#map_data_path="${base_path}/${robot_teacher}/${map_dataset}/manipulabilities_interpolated.csv"
#results_path="${base_path}/${robot_student}/${map_dataset}"

rm -rf ${base_path}/${robot_teacher}/${dataset}/cv
rm -rf ${base_path}/${robot_student}/${dataset}/cv
mkdir -p ${base_path}/${robot_student}/${dataset}/cv
mkdir -p ${base_path}/${robot_teacher}/${dataset}/cv
touch ${base_path}/${robot_student}/${dataset}/cv/errs_naive_icp.txt

# generate artificial data set
if [ generate_data==1 ]; then
	echo "### 	Generate dataset 	###"
	#[ -e ${base_path}/${robot_student}/${dataset}/* ] && rm ${base_path}/${robot_student}/${dataset}/*
	python3 py/generate_artificial_data.py ${base_path} ${dataset} ${dataset} ${volume_scaling} -axes_scaling ${axes_scaling} -cv 1
	cd /home/nnrthmr/CLionProjects/ma_thesis/matlab
	/usr/local/MATLAB/R2021a/bin/matlab -batch "generateRobotDataToy(${base_path@Q},${dataset@Q});exit" | tail -n +17
	cd /home/nnrthmr/CLionProjects/ma_thesis
fi


echo "###	 Cross Validation Initialization	 ###"
python3 py/cv_indices.py ${base_path}/${robot_student}/${dataset}/cv ${dataset} ${k_int}


echo "###	 Cross validate dataset ${map_dataset} from ${robot_teacher} to ${robot_student} (based on dataset ${dataset}) 	###"

for (( i=1; i<=${k_int}; i++ ))
do
	echo "${i}"
	echo "### 	Creating createLookupTable  	###"
	cd /home/nnrthmr/CLionProjects/ma_thesis/matlab
	/usr/local/MATLAB/R2021a/bin/matlab -batch "createLookupTable(${base_path@Q},${robot_teacher@Q},${robot_student@Q},${dataset@Q},${i});exit" | tail -n +17
	cd /home/nnrthmr/CLionProjects/ma_thesis

	# run naive mapping on new data
	echo "### 	Mapping new data with naive lookup table 	###"
	cd /home/nnrthmr/CLionProjects/ma_thesis/matlab
	/usr/local/MATLAB/R2021a/bin/matlab -batch "map_manipulabilities_cv(${base_path@Q},${robot_teacher@Q},${robot_student@Q},${dataset@Q},${i});exit" | tail -n +17
	cd /home/nnrthmr/CLionProjects/ma_thesis

	#run icp on new data
	echo "###	 Running ICP now 	###"
	ip=$((i-1))
	python3 -W ignore py/run_icp.py $base_path $robot_teacher $robot_student $dataset 2 --cv_k ${ip} # 2 because want to find transformation parameters and map new data

	# evaluate fold k
	python3 -W ignore py/cv_evaluate.py $base_path $robot_teacher $robot_student $dataset --cv_k ${ip}

done

# evaluate all folds
python3 -W ignore py/cv_evaluate.py $base_path $robot_teacher $robot_student $dataset




#naive_path="${base_path}/${robot_student}/${map_dataset}/manipulabilities_interpolated_mapped_naive.csv"
#icp_path="${base_path}/${robot_student}/${map_dataset}/manipulabilities_interpolated_mapped_icp.csv"
#input_path="${base_path}/${robot_teacher}/${map_dataset}/manipulabilities_interpolated.csv"
#if [ ${robot_student}=="toy_data" ]; then 
#	ground_truth_path="${base_path}/${robot_student}/${map_dataset}/manipulabilities_interpolated_groundtruth.csv"
#fi

#echo "###	Plotting... 	###"
#python3 py/plot_artificial_data.py ${input_path} ${naive_path} ${icp_path} ${ground_truth_path}
#echo "### 	Done! 	###"
deactivate
	




