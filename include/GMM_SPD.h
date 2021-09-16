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

using namespace Eigen;
using namespace std;

class GMM_SPD {
public:
    int m_k;
    int m_n, m_nData; //number of measured points on trajectory
    int m_nDemos;
    int m_maxIterM, m_maxIterEM;
    int m_dimOutVec; // dim of output in vector form
    int m_dimOut, m_dimIn; //dim of output
    int m_dimVar; //dim of manifold and tangent space
    int m_dimVarVec; //dim of manifold and tangent space in vector form
    int m_dimCovOut; //dim of output covariance
    float m_maxDiffLL;
    float m_regTerm; //regularization term
    float m_dt; //time step duration
    int m_kp; //controller gain
    vector<double> m_priors;
    vector<MatrixXd> m_sigma;
    MatrixXd m_mu, m_muMan;
    MatrixXd m_data;
    MatrixXd m_L, m_gamma, m_H;
    vector<MatrixXd> m_xts;

    GMM_SPD();
    void InitModel(MatrixXd *data);

    MatrixXd getOutOut(MatrixXd m);
    MatrixXd getOutIn(MatrixXd m);
    MatrixXd getInOut(MatrixXd m);
    MatrixXd getInIn(MatrixXd m);

    void EStep();
    void MStep();
    void TrainEM();
    void GMR(MatrixXd *xd, vector<MatrixXd> *sigmaXd);
    VectorXd GaussPDF(MatrixXd data, MatrixXd mu, MatrixXd sig);
    double GaussPDF(double data, double mu, double sig);
    vector<int> linspace(double a, double b, size_t N);
    MatrixXd SPDMean(vector<MatrixXd> mat, int nIter);
    MatrixXd Symmat2Vec(MatrixXd mat);
    vector<MatrixXd> Symmat2Vec(vector<MatrixXd> mat);
    vector<MatrixXd> Vec2Symmat(MatrixXd vec);
    vector<MatrixXd> Vec2Symmat(vector<MatrixXd> vec);
    void CumulativeSum(const VectorXd& input, VectorXd& result);
    vector<MatrixXd> LogmapVec(MatrixXd x, MatrixXd s);
    vector<MatrixXd> LogMap(vector<MatrixXd> X, MatrixXd S);
    vector<MatrixXd> ExpmapVec(MatrixXd x, MatrixXd s);
    vector<MatrixXd> ExpMap(vector<MatrixXd> X, MatrixXd S);
    void SigmaEigenDecomposition(vector<MatrixXd> *sigma, vector<MatrixXd> *V, vector<MatrixXd> *D);
    vector<MatrixXd> ParallelTransport(vector<MatrixXd> S1, vector<MatrixXd> S2);
    MatrixXd ParallelTransport(MatrixXd S1, MatrixXd S2);
};

#endif //MA_THESIS_GMM_SPD_H
