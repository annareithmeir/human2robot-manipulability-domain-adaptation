#ifndef MA_THESIS_UTILS_H
#define MA_THESIS_UTILS_H

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <fstream>

using Eigen::MatrixXd;

std::vector<std::vector<double>> load_csv (const std::string path) {
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<std::vector<double>> values;
    uint rows = 0;

    while (std::getline(indata, line)) {
        if(rows<1){
            ++rows;
            continue;
        }
        std::vector<double> values_i;
        std::stringstream lineStream(line);
        std::string cell;
        uint cols=0;
        while (std::getline(lineStream, cell, ',')) {
            if(cols<3){
                cols++;
                continue;
            }
            values_i.push_back(std::stod(cell));
        }
        values.push_back(values_i);
        ++rows;
    }
    return values;
}

Eigen::Tensor<double, 3> read_manipulabilities(std::string file_path){
    std::vector<std::vector<double>> data = load_csv(file_path);
    Eigen::Tensor<double, 3> manipulabilities(data.size(), 8, 8);
    std::cout<<data.size()<<std::endl;
    manipulabilities.setZero();
    for(int t=0; t<data.size();t++){
        for(int i=0; i<8;i++) {
            for (int j = 0; j < 8; j++) {
                manipulabilities(t, i, j) = data[t][8*i+j];
            }
        }
    }

    return manipulabilities;
}

MatrixXd tensor_product(MatrixXd x, MatrixXd y){

}

MatrixXd tensor_covariance(MatrixXd x){

}

#endif //MA_THESIS_UTILS_H
