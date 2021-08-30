#include "GMM.h"
#include <numeric>
#include "utils.h"

GMM::GMM(std::vector<double> priors, std::vector<Eigen::array<double, 3>> means, std::vector<Matrix3d> covs) {
    this->k = means.size();
    this->means=means; //3x1
    this->covs=covs; //3x3
    this->priors=priors; //double
    assert(accumulate(priors.begin(), priors.end(), 0) == 1.0f);
    assert(k == means.size() == covs.size() == priors.size());
    this->nks=std::vector<double>(k, 0.0);
    this->n = 0;
    this->eps=1e-3;
}

void GMM::e_step() {
    for (int i = 0; i < this->n; i++) {
        vector<float> tmp = this->get_pkxi(x[i]); //returns vector for all k's
        std::transform(this->nks.begin( ), this->nks.end( ), tmp.begin( ), this->nks.begin( ),std::plus<double>( ));
    }
}

void GMM::m_step() {
    for(int k=0; k<this->k; k++){
        vector<double> tmp_mu(this->n,0.0);
        vector<double> tmp_cov(this->n,0.0);
        for (int i = 0; i < this->n; i++) {
            std::vector<double> tmp2 = this->get_pkxi(x[i])[k];
            Tensor<double,3> tmp3 = get_log(x[i], this->means[k]);
            vector<double> tmp2_mu = tmp2 * tmp3;
            vector<double> tmp2_cov = tmp2 * tensor_outer_product(tmp3, tmp3);
            std::transform(tmp_mu.begin( ), tmp_mu.end( ), tmp2_mu.begin( ), tmp_mu.begin( ),std::plus<double>( ));
            std::transform(tmp_cov.begin( ), tmp_cov.end( ), tmp2_cov.begin( ), tmp_cov.begin( ),std::plus<double>( ));
        }
        this->means[k] = (1/this->nks[k]) * get_exp(tmp_mu, this->means[k]);
        this->covs[k] = (1/this->nks[k]) * tmp_cov;
        this->priors[k] = this->nks[k]/this->n;
    }
}

void GMM::train(vector<Tensor<double, 3>> data){
    this->n = data.size(); // ???
    this-> data = data;
    float err = INFINITY;
    while(err > eps){
        e_step();
        m_step(); //set cov and means after loop or in every step?
        //err=???;
    }
}

std::vector<double> GMM::get_pkxi(Tensor<double,3> xi) {
    std::vector<double> pkxi(this->k,0.0);
    for(int k=0; k< this->k; k++)
        pkxi[k] = this->priors[k] * get_normal(xi, this->covs[k], this->means[k]);
    float tmp = accumulate(pkxi.begin(), pkxi.end(), 0);
    for(int k=0; k< this->k; k++)
        pkxi[k] *= 1/tmp;
    return pkxi;
}
