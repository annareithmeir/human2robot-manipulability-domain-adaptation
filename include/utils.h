#ifndef MA_THESIS_UTILS_H
#define MA_THESIS_UTILS_H

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <fstream>
#include <iostream>

#define _USE_MATH_DEFINES

using Eigen::MatrixXd;
using namespace std;
using namespace Eigen;
using Tensor3d = Tensor<double, 3>;

vector<vector<double>> load_csv (const string path) {
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

Tensor3d read_manipulabilities(string file_path){
    vector<vector<double>> data = load_csv(file_path);
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
MatrixXd read_cartesian_trajectories(string file_path){
    vector<vector<double>> data = load_csv(file_path);
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

void load_data(string data_path, int nDemos, int nData, vector<Tensor3d> *data_m, MatrixXd *data_pos){
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

void load_data_cmat(string data_path, MatrixXd *data_pos){
    data_pos->setZero();

    vector<vector<double>> data = load_csv(data_path);
    std::cout<<data.size()<<" "<<data[0].size()<<std::endl;
    for (int t = 0; t < 400; t++) {
        (*data_pos)(0, t) = data[0][t];
        (*data_pos)(1, t) = data[1][t];
        (*data_pos)(2, t) = data[2][t];
    }
}

Tensor<double, 6> tensor_outer_product(Tensor<double, 3> x, Tensor<double, 3> y){
    Eigen::array<IndexPair<long>,0> empty_index_list = {};
    Tensor<double, 6> prod = x.contract(y, empty_index_list);
    return prod;
}

// TODO N is number of trials? --> check if correct
Tensor<double, 6> tensor_covariance(vector<Tensor<double, 3>> x){
    try{
        int N = x.size(); // number of trials
        if(N==1) throw "(N==1) Only one trial provided. Need at least two.";
        Tensor<double, 6> cov(x[0].dimension(0), x[0].dimension(1), x[0].dimension(2),x[0].dimension(0), x[0].dimension(1), x[0].dimension(2));
        cov.setZero();
        for(int i=0; i<N; i++){
            cov=cov+ tensor_outer_product(x[i],x[i]);
        }
        cov=cov* (double) (1/(N-1));
        return cov;
    }
    catch (const char* msg) {
        cerr << msg << endl;
    }
}

// TODO centers around mean of matrix at each time step t --> check if correct
Tensor<double,3> tensor_center(Tensor<double, 3> x){
    Eigen::array<int, 2> dims({1,2});
    Tensor<double, 1> mean = x.mean(dims); //mean of each time step slice
    Tensor<double, 3> c(x.dimension(0), x.dimension(1), x.dimension(2));
    c.setZero();
    for(int t=0; t< c.dimension(0);t++) {
        for(int i=0; i<c.dimension(1);i++) {
            for (int j = 0; j < c.dimension(2); j++) {
                c(t,i,j) = x(t,i,j) - mean(t);
            }
        }
    }
    return c;
}

//Tensor<double,3> get_normal(Tensor<double, 3> x, Tensor<double, 6> cov, Tensor<double,3> m_mu){
//    float d = 8+8*(8-7)/2;
//    vector<Tensor<double,3>> xt = {x};
//    Tensor<double, 3> tmp = (1/pow(2*M_PI, d)) *exp(-0.5* get_log(x, m_mu)*get_inverse(cov)*get_log(x, m_mu));
//}
//
//Tensor<double,3> get_log(Tensor<double,3> x, Tensor<double,3> m_mu){
//    auto tmp = pow(m_mu, 0.5);
//    auto tmp2 = pow(m_mu, -0.5);
//    return tmp * log(tmp2*x*tmp2)*tmp;
//}
//
//Tensor<double,3> get_exp(Tensor<double,3> x,Tensor<double,3> m_mu){
//    auto tmp = pow(m_mu, 0.5);
//    auto tmp2 = pow(m_mu, -0.5);
//    return tmp * exp(tmp2*x*tmp2)*tmp;
//}

//double get_geodesic_distance(Tensor<double,3> a, Tensor<double, 3> b){
//    return 0.0;
//}
//
//double get_approx_geodesic_distance(Tensor<double,3> a, Tensor<double, 3> b){
//    return 0.0;//Stein divergence
//}



#endif //MA_THESIS_UTILS_H
