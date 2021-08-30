#ifndef MA_THESIS_GMM_H
#define MA_THESIS_GMM_H

#include <Eigen/Dense>
#include <vector>
#include <unsupported/Eigen/CXX11/Tensor>


using Eigen::Matrix3d;
using Eigen::VectorXd;

class GMM {
public:
    int k;
    int n;
    float eps;
    std::vector<double> priors;
    std::vector<double> nks;
    std::vector<Matrix3d> covs;
    std::vector<Eigen::array<double, 3>> means;
    std::vector<Eigen::Tensor<double, 3>> data;

    GMM(std::vector<double> priors, std::vector<Eigen::array<double, 3>> means ,std::vector<Matrix3d> covs );
    std::vector<double> get_pkxi(Eigen::Tensor<double,3> xi);
    void e_step();
    void m_step();
    void train(std::vector<Eigen::Tensor<double, 3>> data);

};


#endif //MA_THESIS_GMM_H
