#ifndef MA_THESIS_UTILS_H
#define MA_THESIS_UTILS_H

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/QR>
#include <unsupported/Eigen/CXX11/Tensor>
#include <fstream>
#include <iostream>

#define _USE_MATH_DEFINES
#define deb(x) cout << #x << " " << x << endl;
using namespace std;
using namespace Eigen;
using Tensor3d = Tensor<double, 3>;

inline
vector<vector<double>> LoadCSV (const string path) {
    ifstream indata;
    indata.open(path);
    string line;
    vector<vector<double>> values;
    uint rows = 0;

    while (getline(indata, line)) {
//        if(rows<1){
//            ++rows;
//            continue;
//        }
        vector<double> values_i;
        stringstream lineStream(line);
        string cell;
        uint cols=0;
        while (getline(lineStream, cell, ',')) {
//            if(cols<2){
//                cols++;
//                continue;
//            }
            values_i.push_back(stod(cell));
        }
        values.push_back(values_i);
        ++rows;
    }
    return values;
}

inline
void WriteCSV(const MatrixXd& data, const string path){
    ofstream outdata;
    outdata.open(path);
    outdata << data;
    outdata.close();
}

inline
Tensor3d read_manipulabilities(const string file_path){
    vector<vector<double>> data = LoadCSV(file_path);
    Tensor3d manipulabilities(data.size(), 9, 9);
    manipulabilities.setZero();
    for(int t=0; t<data.size();t++){
        manipulabilities(t, 0, 0) = t * 0.01;
        for(int i=0; i<8;i++) {
            for (int j = 0; j < 8; j++) {
                manipulabilities(t, 1+i, 1+j) = data[t][8*i+j];
            }
        }
    }
    return manipulabilities;
}

//TODO resample spline(), maybe preprocess data by using promp-ias/utils.py interpolation
inline
MatrixXd read_cartesian_trajectories(const string file_path){
    vector<vector<double>> data = LoadCSV(file_path);
    MatrixXd trajs(4,data.size());
    trajs.setZero();
    for (int t = 0; t < data.size(); t++) {
        trajs(0, t) = t*0.01;
        trajs(1, t) = data[t][0];
        trajs(2, t) = data[t][1];
        trajs(3, t) = data[t][2];
    }
    return trajs;
}

inline
void load_data(const string data_path, int nDemos, int nData, vector<Tensor3d> *data_m, MatrixXd *data_pos){
    Tensor3d m;
//    vector<Tensor3d> m_data_m;
    MatrixXd pos;
//    MatrixXd m_data_pos(4, m_nDemos*nData);
    data_pos->setZero();
    string pos_path, m_path;
    for(int i=0;i<nDemos;i++){
        pos_path=data_path + "EEpos_data_trial_"+ to_string(i)+".csv";
        m_path=data_path + "EEpos_manipulability_trial_"+ to_string(i)+".csv";
        m = read_manipulabilities(m_path);
        data_m->push_back(m);

        pos = read_cartesian_trajectories(pos_path);
        data_pos->block(0, i*nData,4, nData) = pos;
    }
}

inline
void load_data_cmat(const string data_path, MatrixXd *data_pos){
    data_pos->setZero();

    vector<vector<double>> data = LoadCSV(data_path);
    std::cout<<data.size()<<" "<<data[0].size()<<std::endl;
    for (int t = 0; t < 400; t++) {
        (*data_pos)(0, t) = data[0][t];
        (*data_pos)(1, t) = data[1][t];
        (*data_pos)(2, t) = data[2][t];
    }
}

inline
void load_data_mmat(const string data_path, MatrixXd *data_m){
    data_m->setZero();

    vector<vector<double>> data = LoadCSV(data_path);
    for (int t = 0; t < 400; t++) {
        (*data_m)(t, 0) = data[t][0];
        (*data_m)(t, 1) = data[t][1];
        (*data_m)(t, 2) = data[t][2];
        (*data_m)(t, 3) = data[t][3];
    }
}

// Checked!
inline
vector<int> linspace(double a, double b, size_t N) {
    double h = (b - a) / static_cast<double>(N - 1);
    vector<int> xs(N);
    vector<int>::iterator x;
    double val;
    for (x = xs.begin(), val = a; x != xs.end(); ++x, val += h) {
        *x = (int) round(val);
    }
    return xs;
}

// Checked!
inline
void CumulativeSum(const VectorXd &input, VectorXd &result) {
    result(0) = input[0];
    for (int i = 1; i < input.size(); i++) {
        result(i) = result(i - 1) + input(i);
    }
}

#endif //MA_THESIS_UTILS_H
