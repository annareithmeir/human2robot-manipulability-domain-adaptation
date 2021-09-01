#ifndef MA_THESIS_GMM_H
#define MA_THESIS_GMM_H

#include <Eigen/Dense>
#include <vector>
#include <unsupported/Eigen/CXX11/Tensor>


using Eigen::MatrixXd;
using Eigen::VectorXd;
using Tensor3d = Eigen::Tensor<double, 3>;

class GMM {
public:
    int m_k;
    int m_n; //number of measured points on trajectory
    int m_nDemos;
    int m_minIterEM, m_maxIterEM;
    int m_dimOutVec; // dim of output in vector form
    int m_dimOut; //dim of output
    int m_dimVar; //dim of manifold and tangent space
    int m_dimVarVec; //dim of manifold and tangent space in vector form
    int m_dimCovOut; //dim of output covariance
    float m_maxDiffLL;
    float m_regTerm; //regularization term
    float m_dt; //time step duration
    int m_kp; //controller gain
    std::vector<double> m_priors;
    std::vector<MatrixXd> m_sigma;
    Eigen::MatrixXd m_mu;
    std::vector<Tensor3d> m_data_m;
    MatrixXd m_data_pos;
    MatrixXd m_L, m_gamma, m_gamma2;

    GMM();
    void InitTrajModel(std::vector<Tensor3d> *data_m, MatrixXd *data_pos);
    void InitManipModel(std::vector<Tensor3d> *data_m, MatrixXd *data_pos);
    void EStep();
    void MStep();
    void TrainEM();
    Eigen::VectorXd GaussPDF(Eigen::MatrixXd mu, Eigen::MatrixXd sig);

};


#endif //MA_THESIS_GMM_H
