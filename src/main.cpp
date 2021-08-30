#include <iostream>
#include <GMM.h>
#include <utils.h>

using namespace std;
using namespace Eigen;
int main() {
//    vector<float> priors = { 0.5, 0.5};
//    MatrixXd cov;
//    cov << 1,2,2,1;
//    vector<MatrixXd> covs = {cov, cov};
//    Eigen::Vector2d m1(1,1);
//    Eigen::Vector2d m2(3,3);
//    vector<VectorXd> means = {m1 , m2};
//    GMM gmm = GMM(2, priors, means, covs);

    string data_path="/home/nnrthmr/Desktop/master-thesis/vrep/vrep_franka_promps/py_scripts/data/EEpos_manipulability_trial_1.csv";
    Tensor<double, 3> m = read_manipulabilities(data_path);
    Tensor<double, 6> mm = tensor_outer_product(m,m);
    tensor_center(m);
    vector<Tensor<double,3>> v = {m,m};
    cout <<tensor_covariance(v);


    return 0;
}
