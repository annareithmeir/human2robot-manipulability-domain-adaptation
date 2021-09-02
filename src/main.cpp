#include <iostream>
#include <GMM_SPD.h>
#include <utils.h>

using namespace std;
using namespace Eigen;
int main() {
    // Load the demonstration data
    //string data_path="/home/nnrthmr/Desktop/master-thesis/vrep/vrep_franka_promps/py_scripts/data/";
//    MatrixXd data_pos(4, 2*21);
//    vector<Tensor3d> data_m;
//    load_data(data_path, 2, 21, &data_m, &data_pos);

//    string cmat_path="/home/nnrthmr/C_Mat.csv";
//    MatrixXd data(3, 400);
//    vector<Tensor3d> data_m;
//    load_data_cmat(cmat_path, &data);

    // Build GMM for trajectories in cartesian coordinates
//
    string mmat_path="/home/nnrthmr/Manip_Mat.csv";
    MatrixXd data(400, 4);
    load_data_mmat(mmat_path, &data);
    GMM_SPD model2 = GMM_SPD();
    model2.InitModel(&data);

    Eigen::MatrixXd d(3, 3);
    d<<1,2,3,4,5,6,7,8,9;
    std::cout<<d<<std::endl;
    std::vector<Eigen::MatrixXd> dd;
    dd.push_back(d);
    dd.push_back(d);
//    model2.CumulativeSum(Eigen::VectorXd::LinSpaced( D,  D-1, 0), id);
    Eigen::MatrixXd m= model2.SPDMean(dd, 1);
    std::cout<<m<<std::endl;
    std::cout<<(d.array().pow(0.5)).matrix()<<std::endl;


    return 0;
}
