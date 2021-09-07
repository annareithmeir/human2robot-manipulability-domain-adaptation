#include <iostream>
#include <GMM_SPD.h>
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


    // Build GMM for trajectories in cartesian coordinates

//    string cmat_path="/home/nnrthmr/C_Mat.csv";
//    MatrixXd data(3, 400);
//    vector<Tensor3d> data_m;
//    load_data_cmat(cmat_path, &data);
//    GMM model = GMM();
//    model.InitModel(&data);
//    model.TrainEM();

    // Build GMM for Manifold

    string mmat_path="/home/nnrthmr/Manip_Mat.csv";
    MatrixXd data(400, 4);
    load_data_mmat(mmat_path, &data);
    GMM_SPD model2 = GMM_SPD();
    model2.InitModel(&data);
    model2.TrainEM();

    return 0;
}