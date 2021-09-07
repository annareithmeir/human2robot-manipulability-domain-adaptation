#ifndef MA_THESIS_GMM_SPD_H
#define MA_THESIS_GMM_SPD_H

#include <Eigen/Dense>
#include <vector>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/MatrixFunctions>
#include <Eigen/Eigenvalues>
#include <numeric>
#include <iostream>
#include <cmath>


using Eigen::MatrixXd;
using Eigen::MatrixXcd;
using Eigen::VectorXd;
using Tensor3d = Eigen::Tensor<double, 3>;

class GMM_SPD {
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
    Eigen::MatrixXd m_mu, m_muMan;
    std::vector<Tensor3d> m_data_m;
    MatrixXd m_data_pos;
    MatrixXd m_L, m_gamma, m_gamma2;
    Tensor3d m_xts;

    GMM_SPD();
    void InitModel(MatrixXd *data);
    void EStep();
    void MStep();
    void TrainEM();
    Eigen::VectorXd GaussPDF(Eigen::MatrixXd mu, Eigen::MatrixXd sig);
    std::vector<int> linspace(double a, double b, std::size_t N);
    Eigen::MatrixXd SPDMean(std::vector<Eigen::MatrixXd> mat, int nIter);
    Eigen::MatrixXd Symmat2Vec(Eigen::MatrixXd mat);
    std::vector<Eigen::MatrixXd> Symmat2Vec(std::vector<Eigen::MatrixXd> mat);
    std::vector<Eigen::MatrixXd> Vec2Symmat(Eigen::MatrixXd vec);
    std::vector<Eigen::MatrixXd> Vec2Symmat(std::vector<Eigen::MatrixXd> vec);
    void CumulativeSum(const Eigen::VectorXd& input, Eigen::VectorXd& result);
    std::vector<MatrixXd> LogmapVec(MatrixXd x, MatrixXd s);
    std::vector<MatrixXd> LogMap(std::vector<MatrixXd> X, MatrixXd S);

};


#endif //MA_THESIS_GMM_SPD_H
