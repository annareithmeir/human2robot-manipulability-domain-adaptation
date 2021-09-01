#include <iostream>
#include <GMM.h>
#include <utils.h>

using namespace std;
using namespace Eigen;
int main() {
    // Load the demonstration data
    //string data_path="/home/nnrthmr/Desktop/master-thesis/vrep/vrep_franka_promps/py_scripts/data/";
//    MatrixXd data_pos(4, 2*21);
//    vector<Tensor3d> data_m;
//    load_data(data_path, 2, 21, &data_m, &data_pos);

    string cmat_path="/home/nnrthmr/C_Mat.csv";
    MatrixXd data(3, 400);
    vector<Tensor3d> data_m;
    load_data_cmat(cmat_path, &data);

    // Build GMM for trajectories in cartesian coordinates
    GMM model1 = GMM();
    model1.InitTrajModel(&data_m, &data);
    model1.TrainEM();



    return 0;
}
