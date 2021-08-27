#include "GMM.h"


GMM::GMM(int k, std::vector<float> priors, std::vector<VectorXd> means, std::vector<MatrixXd> covs) {
    this->k = k;
    this->means=means;
    this->covs=covs;
}

void GMM::e_step() {

}

void GMM::m_step() {

}

SPD GMM::get_px(SPD x) {
    return x;
}

SPD GMM::get_pkxi(SPD x, int k, int i) {
    return SPD();
}
