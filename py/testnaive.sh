#!/bin/bash
set -e

source venv3-6/bin/activate

input_path="/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/rhuman/sing_up_60/manipulabilities.csv"
icp_path="/home/nnrthmr/CLionProjects/ma_thesis/data/final_results/naive/sing_up_60/manipulabilities_interpolated_mapped_naive_500.csv" # human to robot
ground_truth_path="/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/panda/sing_up_60/manipulabilities.csv"
#paths_list="${input_path},${ground_truth_path}"
paths_list="${input_path},${ground_truth_path},${icp_path}"
echo "###	Plotting 	###"
python3 plot_artificial_data.py -mapping_paths ${paths_list}