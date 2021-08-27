#ifndef MA_THESIS_GMM_H
#define MA_THESIS_GMM_H

#include <Eigen/Dense>
#include <vector>
#include <SPD.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class GMM {
public:
    int k;
    std::vector<float> priors;
    std::vector<float> nks;
    std::vector<MatrixXd> covs;
    std::vector<VectorXd> means;

    GMM(int k, std::vector<float> priors, std::vector<VectorXd> means ,std::vector<MatrixXd> covs );
    SPD get_px(SPD x);
    SPD get_pkxi(SPD x, int k, int i);
    void e_step();
    void m_step();
};


#endif //MA_THESIS_GMM_H
