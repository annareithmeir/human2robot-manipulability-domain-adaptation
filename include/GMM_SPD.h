#ifndef MA_THESIS_GMM_SPD_H
#define MA_THESIS_GMM_SPD_H

#include <vector>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/MatrixFunctions>
#include <Eigen/Eigenvalues>
#include <numeric>
#include <cmath>
#include "SPD_Utils.h"

using namespace Eigen;
using namespace std;

class GMM_SPD {
public:
    int m_k;
    int m_n, m_nData; //number of measured points on trajectory
    int m_nDemos;
    int m_maxIterM, m_maxIterEM, m_minIterEM;
    int m_dimOutVec; // dim of output in vector form
    int m_dimOut, m_dimIn; //dim of output
    int m_dimVar; //dim of manifold and tangent space
    int m_dimVarVec; //dim of manifold and tangent space in vector form
    int m_dimCovOut; //dim of output covariance
    float m_maxDiffLL;
    float m_regTerm; //regularization term
    float m_dt; //time step duration
    int m_kp; //controller gain
    int m_km; //manipulability gain
    vector<double> m_priors;
    vector<MatrixXd> m_sigma;
    MatrixXd m_mu, m_muMan;
    MatrixXd m_data;
    MatrixXd m_L, m_gamma, m_H;
    vector<MatrixXd> m_xts;

    GMM_SPD();
    void InitModel(const MatrixXd& data);

    MatrixXd getOutOut(const MatrixXd& m);
    MatrixXd getOutIn(const MatrixXd& m);
    MatrixXd getInOut(const MatrixXd& m);
    MatrixXd getInIn(const MatrixXd& m);

    void EStep();
    void MStep();
    void TrainEM();
    void GMR(MatrixXd& xd, vector<MatrixXd>& sigmaXd);
    void GMR(MatrixXd& xd, vector<MatrixXd>& sigmaXd, int t);
    VectorXd GaussPDF(const MatrixXd& data, const MatrixXd& mu, const MatrixXd& sig);
    double GaussPDF(double data, double mu, double sig);
    void SigmaEigenDecomposition(const vector<MatrixXd>& sigma, vector<MatrixXd>& V, vector<MatrixXd>& D);
};

#endif //MA_THESIS_GMM_SPD_H
