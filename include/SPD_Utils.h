#ifndef MA_THESIS_SPD_UTILS_H
#define MA_THESIS_SPD_UTILS_H

#include "utils.h"
#include <unsupported/Eigen/MatrixFunctions>

using namespace std;
using namespace Eigen;


// Checked!
inline
MatrixXd ParallelTransport(const MatrixXd& S1, const MatrixXd& S2) {
    MatrixXd S3;
    S3 = (S1.transpose().inverse() * S2.transpose()).transpose().pow(0.5); //B/A = (A'\B')' and A\B = A.inv()*B
    return S3;
}

// Checked!
inline
vector<MatrixXd> ParallelTransport(const vector<MatrixXd>& S1, const vector<MatrixXd>& S2) {
    vector<MatrixXd> S3;
    for (int i = 0; i < S1.size(); i++) {
        S3.push_back(ParallelTransport(S1[i], S2[i])); //B/A = (A'\B')' and A\B = A.inv()*B
    }
    return S3;
}

// Checked!
inline
MatrixXd Symmat2Vec(const MatrixXd& mat) {
    int N = mat.rows();
    vector<double> v;
    VectorXd dia = mat.diagonal();
    for (int x = 0; x < dia.size(); x++) {
        v.push_back(dia(x));
    }
    int row, col;
    for (int n = 1; n < N; n++) {
        row = 0;
        col = n;
        for (int ni = n; ni < N; ni++) {
            v.push_back(sqrt(2) * mat(row, col));
            row++;
            col++;
        }
    }
    MatrixXd vm(1, v.size());
    for (int x = 0; x < v.size(); x++) {
        vm(0, x) = v[x]; //one row
    }
    return vm;
}

// Checked!
inline
vector<MatrixXd> Symmat2Vec(const vector<MatrixXd>& mat_vec) {
    MatrixXd vn;
    int N = mat_vec.size();
    vector<MatrixXd> vec;
    for (int i = 0; i < N; i++) {
        vn = Symmat2Vec(mat_vec[i]);
        vec.push_back(vn);
    }
    return vec;
}

// Checked!
inline
vector<MatrixXd> Vec2Symmat(const MatrixXd& vec) {
    vector<MatrixXd> MVector;
    MatrixXd vn, Mn;
    int d = vec.rows();
    int N = vec.cols();
    int D = (-1 + sqrt(1 + 8 * d)) / (double) 2;
    VectorXd id(D);
    int row;
    for (int n = 0; n < N; n++) { //colwise
        vn = vec.col(n).transpose();
        Mn = vn.row(0).leftCols(D).asDiagonal();
        id.setZero();
        CumulativeSum(VectorXd::LinSpaced(D, D, 1), id);
        MatrixXd tmp1(Mn.rows(), Mn.cols());
        MatrixXd tmp2(Mn.rows(), Mn.cols());
        for (int i = 1; i < D; i++) {
            tmp1.setZero();
            row = 0;
            for (int k = i; k < id(i) - id(i - 1) + i; k++) {
                tmp1(row, k) = vn(0, id(i - 1) + row) * (1 / sqrt(2));
                row++;
            }
            tmp2.setZero();
            row = 0;
            for (int k = i; k < id(i) - id(i - 1) + i; k++) {
                tmp2(k, row) = vn(0, id(i - 1) + row) * (1 / sqrt(2));
                row++;
            }
            Mn = Mn + tmp1 + tmp2;
        }
        MVector.push_back(Mn);
    }
    return MVector;
}

// Checked!
inline
vector<MatrixXd> Vec2Symmat(const vector<MatrixXd>& vec) {
    MatrixXd v(vec[0].rows(), vec.size());
    for (int i = 0; i < vec.size(); i++) {
        v.col(i) = vec[i];
    }
    return Vec2Symmat(v);
}

// Checked!
inline
vector<MatrixXd> ExpMap(const vector<MatrixXd>& U, const MatrixXd& S) {
    vector<MatrixXd> X;
    MatrixXd D,V, tmp2;
    for (int i = 0; i < U.size(); i++) {
        MatrixXd tmp = (S.inverse()) * U[i]; //A\B in MATLAB is a^-1 * B
        EigenSolver<MatrixXd> es(tmp);
        D = es.eigenvalues().real().asDiagonal();
        V = es.eigenvectors().real();
        tmp2 = D.diagonal().array().exp().matrix().asDiagonal().toDenseMatrix();
        X.push_back(S * V * tmp2 * V.inverse());
    }
    return X;
}

// Checked!
inline
vector<MatrixXd> ExpmapVec(const MatrixXd& u, const MatrixXd& s) {
    vector<MatrixXd> U = Vec2Symmat(u);
    vector<MatrixXd> S = Vec2Symmat(s);
    vector<MatrixXd> X = ExpMap(U, S[0]); //Vec2Symmat gives back vector of size 1 here
    vector<MatrixXd> x = Symmat2Vec(X);
    return x;
}

// Checked!
inline
vector<MatrixXd> LogMap(const vector<MatrixXd>& X, const MatrixXd& S) {
    vector<MatrixXd> U;
    MatrixXd tmp, D, V, tmp2;
    for (int i = 0; i < X.size(); i++) {
        tmp = (S.inverse()) * X[i]; //A\B in MATLAB is a^-1 * B
        EigenSolver<MatrixXd> es(tmp);
        D = es.eigenvalues().real().asDiagonal();
        V = es.eigenvectors().real();
        tmp2 = D.diagonal().array().log().matrix().asDiagonal().toDenseMatrix();
        U.push_back(S * V * tmp2 * V.inverse());
    }
    return U;
}

inline
MatrixXd LogMap(const MatrixXd& X, const MatrixXd& S) {
    MatrixXd U;
    MatrixXd tmp = (S.inverse()) * X; //A\B in MATLAB is a^-1 * B
    EigenSolver<MatrixXd> es(tmp);
    MatrixXd D = es.eigenvalues().real().asDiagonal();
    MatrixXd V = es.eigenvectors().real();
    MatrixXd tmp2 = D.diagonal().array().log().matrix().asDiagonal().toDenseMatrix();
    U = S * V * tmp2 * V.inverse();
    return U;
}

// Checked!
inline
vector<MatrixXd> LogmapVec(const MatrixXd& x, const MatrixXd& s) {
    vector<MatrixXd> X = Vec2Symmat(x);
    vector<MatrixXd> S = Vec2Symmat(s);
    vector<MatrixXd> U = LogMap(X, S[0]); //Vec2Symmat gives back vector of size 1 here
    vector<MatrixXd> u = Symmat2Vec(U);
    return u;
}

// Checked!
inline
MatrixXd SPDMean(const vector<MatrixXd>& mat, int nIter) {
    MatrixXd M = mat[0];
    MatrixXd tmp;
    MatrixXd L(mat[0].rows(), mat[0].cols());
    for (int iter = 0; iter < nIter; iter++) {
        L.setZero();
        for (int i = 0; i < mat.size(); i++) {
            tmp = M.pow(-0.5) * mat[i] * (M.pow(-0.5));
            L = L + (tmp.log()).matrix();
        }
        M = M.pow(0.5) * (L.array() / mat.size()).matrix().exp().matrix() * M.pow(0.5);
    }
    return M;
}


#endif //MA_THESIS_SPD_UTILS_H
