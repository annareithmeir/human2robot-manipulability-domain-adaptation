#ifndef MA_THESIS_GMM_H
#define MA_THESIS_GMM_H

#include <vector>
#include <numeric>
#include <cmath>
#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

class GMM {
public:
    int m_k;
    int m_n, m_nData; //number of measured points on trajectory
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
    MatrixXd m_data_pos;
    MatrixXd m_L, m_gamma, m_gamma2;

    GMM();
    void InitModel(const MatrixXd& data, int demos);
    void EStep();
    void MStep();
    void TrainEM();
    void GMR(MatrixXd& xd, vector<MatrixXd>& sigmaXd);
    void GMR(MatrixXd& xd, vector<MatrixXd>& sigmaXd, int t);
    Eigen::VectorXd GaussPDF(const MatrixXd& mu, const MatrixXd& sig);
    double GaussPDF(double data, double mu, double sig);
    std::vector<double> linspace(double a, double b, std::size_t N);
};


#endif //MA_THESIS_GMM_H
