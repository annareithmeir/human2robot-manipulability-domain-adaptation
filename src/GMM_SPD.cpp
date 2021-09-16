#include "GMM_SPD.h"
#include <Eigen/Core>

GMM_SPD::GMM_SPD() {
    this->m_k = 5;
    this->m_n = -1;
    this->m_maxDiffLL = 1e-4; //Likelihood increase threshold to stop algorithm
    this->m_maxIterEM = 1;
    this->m_maxIterM = 10;

    this->m_dimVar = 4;
    this->m_dimIn = 1;
    this->m_dimOut = 3;

//    this->m_dimVar = 3;
//    this->m_dimIn = 1;
//    this->m_dimOut = 2;

    this->m_dimOutVec = 2 + 2 * (2 - 1) / 2;
    this->m_dimVarVec = 3 - 2 + this->m_dimOutVec;
    this->m_dimCovOut = 3 + 3 * (3 - 1) / 2;

//    this->m_dimOutVec = this->m_dimOut + this->m_dimOut * (this->m_dimOut - 1) / 2;
//    this->m_dimVarVec = this->m_dimVar - this->m_dimOut + this->m_dimOutVec;
//    this->m_dimCovOut = this->m_dimVar + this->m_dimVar * (this->m_dimVar - 1) / 2;
    this->m_dt = 1e-2;
    this->m_regTerm = 1e-4;
    this->m_kp = 100;
    this->m_nDemos = 4;
}

MatrixXd GMM_SPD::getInOut(MatrixXd m) {
    return m.block(0, this->m_dimIn, this->m_dimIn, this->m_dimOut);
}

MatrixXd GMM_SPD::getInIn(MatrixXd m) {
    return m.block(0, 0, this->m_dimIn, this->m_dimIn);
}

MatrixXd GMM_SPD::getOutIn(MatrixXd m) {
    return m.block(this->m_dimIn, 0, this->m_dimOut, this->m_dimIn);
}

MatrixXd GMM_SPD::getOutOut(MatrixXd m) {
    return m.block(this->m_dimIn, this->m_dimIn, this->m_dimOut, this->m_dimOut);
}

// Checked!
vector<int> GMM_SPD::linspace(double a, double b, size_t N) {
    double h = (b - a) / static_cast<double>(N - 1);
    vector<int> xs(N);
    vector<int>::iterator x;
    double val;
    for (x = xs.begin(), val = a; x != xs.end(); ++x, val += h) {
        *x = (int) round(val);
    }
    return xs;
}

// Checked!
void GMM_SPD::CumulativeSum(const VectorXd &input, VectorXd &result) {
    result(0) = input[0];
    for (int i = 1; i < input.size(); i++) {
        result(i) = result(i - 1) + input(i);
    }
}

//Checked!
void GMM_SPD::InitModel(MatrixXd *data) {
    this->m_n = data->rows() / this->m_nDemos; //nbData
    this->m_nData = data->rows(); //nb
    this->m_muMan = MatrixXd(4, this->m_k);
    this->m_mu = MatrixXd(this->m_muMan.rows(), this->m_muMan.cols());
    this->m_mu.setZero();
    this->m_data = *data;
    this->m_muMan.setZero();
    this->m_muMan << 0.105000000000000, 0.305000000000000, 0.505000000000000, 0.705000000000000, 0.905000000000000,
            146.096605076556, 50.2725428695640, 57.5434325637446, 61.0162279884024, 48.6986348781330,
            44.2213672844795, 70.3227237095515, 52.9208375483169, 43.8637203321041, 176.322381110539,
            -71.9140893269118, 33.7271691014874, -43.1366974235137, 45.0107695266492, 106.855203790622;

    vector<int> timing = linspace(0, this->m_n, this->m_k + 1);
    for (int i = 0; i < this->m_k; i++) {
        vector<int> collected;
        for (int d = 0; d < this->m_nDemos; d++) {
            for (int t = timing[i]; t < timing[i + 1]; t++) {
                collected.push_back(d * this->m_n + t);
            }
        }
        this->m_priors.push_back(collected.size());
        MatrixXd collectedMatrix(collected.size(), 3);
        MatrixXd collectedMatrixFull(collected.size(), 4); // with time
        for (int l = 0; l < collected.size(); l++) {
            collectedMatrix.block(l, 0, 1, 3) = (*data).row(collected[l]).rightCols(3); // collected matrices checked!
            collectedMatrixFull.row(l) = (*data).row(collected[l]);
        }

        //TODO Find out why small numerical errors to MATLAB code here (in range 1e-03)

        // MuMan checked!
//        this->m_muMan.col(i) = collectedMatrixFull.colwise().mean();
//        this->m_muMan.block(1,i,3,1) = Symmat2Vec(SPDMean(Vec2Symmat(collectedMatrix.transpose()), 10)).transpose();

        // DataTangent checked!
        vector<MatrixXd> dataTangent = LogmapVec(collectedMatrix.transpose(),
                                                      this->m_muMan.col(i).bottomRows(3)); // cut off t data
        MatrixXd dataTangentMatrix(4, dataTangent.size());
        dataTangentMatrix.setZero();
        for (int i = 0; i < dataTangent.size(); i++) {
            dataTangentMatrix(0, i) = (*data)(collected[i], 0);
            dataTangentMatrix.block(1, i, 3, 1) = dataTangent[i].transpose(); //to matrix as in matlab code, with t row
        }

        // Cov calculation checked!
        MatrixXd centeredTangent = dataTangentMatrix.colwise() - dataTangentMatrix.rowwise().mean();
//        MatrixXd cov = (centeredTangent * centeredTangent.adjoint()) / double(dataTangentMatrix.cols() - 1);
        MatrixXd cov = (centeredTangent * centeredTangent.adjoint()) / double(dataTangentMatrix.cols() - 1) +
                       MatrixXd(this->m_dimVarVec, this->m_dimVarVec).setConstant(this->m_regTerm);
        this->m_sigma.push_back(cov);
    }

    // Priors checked!
    double priorsSum = (double) (accumulate(this->m_priors.begin(), this->m_priors.end(), 0.0f));
    for (double &d: this->m_priors) {
        d /= priorsSum;
    }

    this->m_L = MatrixXd(this->m_k, data->rows());
    this->m_L.setZero();
    this->m_gamma = MatrixXd(this->m_k, data->cols());
    this->m_H = MatrixXd(this->m_k, data->cols());

//    cout << "\n\n\nAFTER INITIALISATION\n" << endl;
//    cout << "muMan= \n" << this->m_muMan << endl; // error of up to 0.3
//    for(int k=0; k<this->m_k; k++) {
//        cout << "sigma= \n" << this->m_sigma[k] << endl;
//        cout << "prior= \n" << this->m_priors[k] << endl;
//    }
}


// Checked!
MatrixXd GMM_SPD::SPDMean(vector<MatrixXd> mat, int nIter) {
    MatrixXd M = mat[0];
    for (int iter = 0; iter < nIter; iter++) {
        MatrixXd L(mat[0].rows(), mat[0].cols());
        L.setZero();
        for (int i = 0; i < mat.size(); i++) {
            MatrixXd tmp = M.pow(-0.5) * mat[i] * (M.pow(-0.5));
            L = L + (tmp.log()).matrix();
        }
        M = M.pow(0.5) * (L.array() / mat.size()).matrix().exp().matrix() * M.pow(0.5);
    }
    return M;
}

// Checked!
MatrixXd GMM_SPD::Symmat2Vec(MatrixXd mat) {
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
vector<MatrixXd> GMM_SPD::Symmat2Vec(vector<MatrixXd> mat_vec) {
    int N = mat_vec.size();
    vector<MatrixXd> vec;
    for (int i = 0; i < N; i++) {
        MatrixXd vn = Symmat2Vec(mat_vec[i]);
        vec.push_back(vn);
    }
    return vec;
}

// Checked!
vector<MatrixXd> GMM_SPD::Vec2Symmat(MatrixXd vec) {
    vector<MatrixXd> MVector;
    int d = vec.rows();
    int N = vec.cols();
    int D = (-1 + sqrt(1 + 8 * d)) / (double) 2;
    int row;
    for (int n = 0; n < N; n++) { //colwise
        MatrixXd vn = vec.col(n).transpose();
        MatrixXd Mn = vn.row(0).leftCols(D).asDiagonal();
        VectorXd id(D);
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
vector<MatrixXd> GMM_SPD::Vec2Symmat(vector<MatrixXd> vec) {
    MatrixXd v(vec[0].rows(), vec.size());
    for (int i = 0; i < vec.size(); i++) {
        v.col(i) = vec[i];
    }
    return Vec2Symmat(v);
}

// Checked!
vector<MatrixXd> GMM_SPD::LogmapVec(MatrixXd x, MatrixXd s) {
    vector<MatrixXd> X = Vec2Symmat(x);
//    cout<<"\n\n\nIN LOGMAP VEC (vec2symmat)\n"<<endl;
//    cout<<"x: \n"<<x<<endl;
//    cout<<"X: \n"<<X[0]<<endl;
//    cout<<"\n\n\nEND LOGMAP VEC\n"<<endl;
    vector<MatrixXd> S = Vec2Symmat(s);
    vector<MatrixXd> U = LogMap(X, S[0]); //Vec2Symmat gives back vector of size 1 here
    vector<MatrixXd> u = Symmat2Vec(U);
    return u;
}

// Checked!
vector<MatrixXd> GMM_SPD::LogMap(vector<MatrixXd> X, MatrixXd S) {
    vector<MatrixXd> U;
    for (int i = 0; i < X.size(); i++) {
        MatrixXd tmp = (S.inverse()) * X[i]; //A\B in MATLAB is a^-1 * B
        EigenSolver<MatrixXd> es(tmp);
        MatrixXd D = es.eigenvalues().real().asDiagonal();
        MatrixXd V = es.eigenvectors().real();
//        cout<<"tmp:\n";
//        cout<<tmp<<endl;
//        cout<<"D:\n";
//        cout<<D<<endl;
//        cout<<"V:\n";
//        cout<<V<<endl;
//        cout<<"\nres:\n";
//        cout<<(D.diagonal());
//        cout<<"\nres2:\n";
        MatrixXd tmp2 = D.diagonal().array().log().matrix().asDiagonal().toDenseMatrix();
//        cout<<tmp2;
//        cout<<"\nres3:\n";
//        cout<<S*V*tmp2*V.inverse()<<endl;
        U.push_back(S * V * tmp2 * V.inverse());
    }
    return U;
}

//Checked! (numerically slightly different)
VectorXd GMM_SPD::GaussPDF(MatrixXd data, MatrixXd mu, MatrixXd sig) {
    MatrixXd pdf(1, this->m_n);
    MatrixXd dataCentered = data.transpose() - mu.transpose().replicate(data.cols(), 1);
    MatrixXd tmp = dataCentered * (sig.inverse());
    pdf = (tmp.array() * dataCentered.array()).matrix().rowwise().sum();
    pdf = (-0.5 * pdf).array().exp() /
          sqrt(pow(2 * M_PI, this->m_dimVar) * abs(sig.determinant()) + numeric_limits<double>::min());
    return pdf;
}

double GMM_SPD::GaussPDF(double data, double mu, double sig) {
    double pdf;
    double dataCentered = data - mu;
    double tmp = dataCentered / sig;
//    cout<<"\n\n IN GAUSS\n"<<endl;
//    cout<<"\ndata\n"<<data<<endl;
//    cout<<"\nmu\n"<<mu<<endl;
//    cout<<"\nsig\n"<<sig<<endl;
//
//    cout<<"\ndatacentered\n"<<dataCentered<<endl;
//    cout<<"\ntmp\n"<<tmp<<endl;
    pdf = tmp * dataCentered;
//    cout<<"\npdf\n"<<pdf<<endl;
    pdf = exp(-0.5 * pdf) / sqrt(2 * M_PI * abs(sig) + numeric_limits<double>::min());
//    cout<<"\npdf\n"<<pdf<<endl;
    return pdf;
}

// Checked -> small numerical errors
void GMM_SPD::EStep() {
    for (int k = 0; k < this->m_k; k++) {
        MatrixXd tmp(1, this->m_n * m_nDemos);
        tmp.setConstant(this->m_muMan(0, k));
        MatrixXd xts(this->m_dimVarVec, this->m_data.rows());
        xts.setZero();
        xts.row(0) = this->m_data.col(0).transpose() - tmp;

        vector<MatrixXd> logmapvec = LogmapVec(this->m_data.transpose().bottomRows(3),
                                                    this->m_muMan.block(1, k, 3, 1)); //row vector
        for (int v = 0; v < logmapvec.size(); v++) {
            xts.block(1, v, 3, 1) = logmapvec[v].transpose();
        }
        this->m_xts.push_back(xts);
//        cout<<"prior: \n"<<this->m_priors[k]<<endl;
//        cout<<"HERE"<<endl;
//        cout<<"this->m_xts[k].transpose(): \n"<<this->m_xts[k].transpose()<<endl; // already here instabilities
//        cout<<"this->m_mu: \n"<<this->m_mu.col(k)<<endl;
//        cout<<"this->m_sigma[k]: \n"<<this->m_sigma[k]<<endl;
//        cout<<"GaussPDF(this->m_xts[k], this->m_mu.col(k), this->m_sigma[k]): \n"<<GaussPDF(this->m_xts[k], this->m_mu.col(k), this->m_sigma[k])<<endl;
        this->m_L.row(k) =
                this->m_priors[k] * GaussPDF(this->m_xts[k], this->m_mu.col(k), this->m_sigma[k]).transpose();
    }
//    this->m_gamma = (this->m_L.array() / (this->m_L.colwise().sum().array()).replicate(this->m_k, 1).array()).matrix();
    this->m_gamma = (this->m_L.array() /
                     (this->m_L.colwise().sum().array() + numeric_limits<double>::min()).replicate(this->m_k,
                                                                                                        1).array()).matrix();
//    this->m_H = (this->m_gamma.array() / (this->m_gamma.rowwise().sum().array()).replicate(1,this->m_n*this->m_nDemos).array()).matrix();
    this->m_H = (this->m_gamma.array() /
                 (this->m_gamma.rowwise().sum().array() + numeric_limits<double>::min()).replicate(1, this->m_n *
                                                                                                           this->m_nDemos).array()).matrix();

}

//Checked!
void GMM_SPD::MStep() {
    for (int k = 0; k < this->m_k; k++) {
        this->m_priors[k] = this->m_gamma.row(k).sum() / this->m_data.rows();
        MatrixXd uTmp(this->m_dimVarVec, this->m_data.rows());
        for (int n = 0; n < this->m_maxIterM; n++) {

            //Upd on tangent space
            uTmp.setZero();
            MatrixXd tmp(1, this->m_n * m_nDemos);
            tmp.setConstant(this->m_muMan(0, k));
            uTmp.row(0) = this->m_data.col(0).transpose() - tmp;
            vector<MatrixXd> logmapvec = LogmapVec(this->m_data.transpose().bottomRows(3),
                                                        this->m_muMan.block(1, k, 3, 1)); //row vector
            for (int v = 0; v < logmapvec.size(); v++) {
                uTmp.block(1, v, 3, 1) = logmapvec[v].transpose();
            }
            MatrixXd uTmpTot = (uTmp.array() *
                                (this->m_H.row(k).replicate(this->m_dimVarVec, 1)).array()).matrix().rowwise().sum();

            //Upd on manifold
            this->m_muMan(0, k) = uTmpTot(0) + this->m_muMan(0, k);
            this->m_muMan.block(1, k, 3, 1) = ExpmapVec(uTmpTot.bottomRows(3),
                                                        this->m_muMan.block(1, k, 3, 1))[0].transpose();
        }
        this->m_sigma[k] = (uTmp * this->m_H.row(k).asDiagonal() * uTmp.transpose()) +
                           (MatrixXd(this->m_dimVarVec, this->m_dimVarVec).setIdentity() * this->m_regTerm);
    }

//    cout << "\n\n\nAFTER EM ALGORITHM\n" << endl;
//    cout << "muMan= \n" << this->m_muMan << endl;
    // If niter 10 and niterem 1 error same, if niter 1 and niterem 10 then absolutely high -> something happens in between? TODO check!
    // error up to 0.3, when using exact initialized muman, then only up to 0.02 (1,1)
    // when using exact initialized muman, up to 0.03 (10,1)
    // when using exact initialized muman, HUGE! (10,10)
//    for(int k=0; k<this->m_k; k++) {
//        cout << "sigma= \n" << this->m_sigma[k] << endl;
//        cout << "prior= \n" << this->m_priors[k] << endl;
//    }
}

void GMM_SPD::TrainEM() {
    for (int iter = 0; iter < this->m_maxIterEM; iter++) {
        cout << "EM iteration #" << iter << " ";
        EStep();
        cout << "e step done ";
        MStep();
        cout << "m step done" << endl;
    }
    cout << " The maximum number of iterations has been reached." << endl;
}

// Checked!
vector<MatrixXd> GMM_SPD::ExpmapVec(MatrixXd u, MatrixXd s) {
    vector<MatrixXd> U = Vec2Symmat(u);
    vector<MatrixXd> S = Vec2Symmat(s);
    vector<MatrixXd> X = ExpMap(U, S[0]); //Vec2Symmat gives back vector of size 1 here
    vector<MatrixXd> x = Symmat2Vec(X);
    return x;
}

// Checked!
vector<MatrixXd> GMM_SPD::ExpMap(vector<MatrixXd> U, MatrixXd S) {
    vector<MatrixXd> X;
    for (int i = 0; i < U.size(); i++) {
        MatrixXd tmp = (S.inverse()) * U[i]; //A\B in MATLAB is a^-1 * B
        EigenSolver<MatrixXd> es(tmp);
        MatrixXd D = es.eigenvalues().real().asDiagonal();
        MatrixXd V = es.eigenvectors().real();
        MatrixXd tmp2 = D.diagonal().array().exp().matrix().asDiagonal().toDenseMatrix();
        X.push_back(S * V * tmp2 * V.inverse());
    }
    return X;
}

void
GMM_SPD::SigmaEigenDecomposition(vector<MatrixXd> *Sigma, vector<MatrixXd> *V, vector<MatrixXd> *D) {
    for (int i = 0; i < Sigma->size(); i++) {
//        cout<<"in eig() \n"<<(*Sigma)[i]<<endl;
        EigenSolver<MatrixXd> es((*Sigma)[i]);
        D->push_back(es.eigenvalues().real().asDiagonal());
        V->push_back(es.eigenvectors().real());
//        cout<<"tmp:\n";
//        cout<<tmp<<endl;
//        cout<<"D:\n";
//        cout<<D<<endl;
//        cout<<"V:\n";
//        cout<<V<<endl;
//        cout<<"\nres:\n";
//        cout<<(D.diagonal());
//        cout<<"\nres2:\n";

    }
}

vector<MatrixXd> GMM_SPD::ParallelTransport(vector<MatrixXd> S1, vector<MatrixXd> S2) {
    vector<MatrixXd> S3;
    for (int i = 0; i < S1.size(); i++) {
        S3.push_back(ParallelTransport(S1[i], S2[i])); //B/A = (A'\B')' and A\B = A.inv()*B
    }
    return S3;
}

MatrixXd GMM_SPD::ParallelTransport(MatrixXd S1, MatrixXd S2) {
    MatrixXd S3;
//    cout<<" IN PARALLEL"<<endl;
//    cout<<(S1.transpose().inverse() * S2.transpose()).transpose()<<endl;
//    cout<<" IN PARALLEL"<<endl;
//    cout<<(S1.transpose().inverse() * S2.transpose()).transpose().pow(0.5)<<endl;
    S3 = (S1.transpose().inverse() * S2.transpose()).transpose().pow(0.5); //B/A = (A'\B')' and A\B = A.inv()*B
    return S3;
}

void GMM_SPD::GMR(MatrixXd *xd, vector<MatrixXd> *sigmaXd) {
    MatrixXd xIn = VectorXd::LinSpaced(this->m_n, this->m_dt,
                                              this->m_n * this->m_dt).matrix().transpose(); // 1x100
    MatrixXd uHat(3, this->m_n);
    MatrixXd xHat(3, this->m_n);
    MatrixXd Ac(3, 3);
    Ac.setZero();
    vector<MatrixXd> S1;
    vector<MatrixXd> S2;
    vector<MatrixXd> uOut;  //3x5x400
    vector<MatrixXd> expSigma;  //3x3x400
    MatrixXd expSigmaT(3, 3);
    MatrixXd H(this->m_k, this->m_n);
    H.setZero();
    uHat.setZero();
    xHat.setZero();
    MatrixXf::Index max_index;
    MatrixXd SigmaOutTmp;

    vector<MatrixXd> pvMatTmp;

    vector<MatrixXd> V;
    vector<MatrixXd> D;
    SigmaEigenDecomposition(&this->m_sigma, &V, &D);

    vector<vector<MatrixXd>> vMat;
    vector<vector<MatrixXd>> pvMat;
    vector<MatrixXd> pvMatK;
    vector<MatrixXd> vMatK;
    vector<MatrixXd> pv;
    vector<MatrixXd> pSigma;

    MatrixXd uoutTmp(3, this->m_k);
    MatrixXd vMatTmp(3, 3);
    MatrixXd pvKTmp(4, 4);

    for (int t = 0; t < this->m_n; t++) {
        for (int k = 0; k < this->m_k; k++) {
            H(k, t) = this->m_priors[k] *
                      GaussPDF(xIn(0, t) - this->m_muMan(0, k), this->m_mu(0, k), this->m_sigma[k](0, 0));
        }
        H.col(t) = (H.col(t).array() / (H.col(t).array() + numeric_limits<double>::min()).colwise().sum()(0,
                                                                                                               0)).matrix(); // sum only one value here

        // Compute conditional mean (with covariance transportation)
        if (t == 0) {
            H.col(t).maxCoeff(&max_index);
            xHat.col(t) = this->m_muMan.block(1, max_index, 3, 1);
        } else {
            xHat.col(t) = xHat.col(t - 1);
        }

        // Iterative computation
        for (int iter = 0; iter < 10; iter++) {
            uHat.col(t).setZero();
            uoutTmp.setZero();

            pv.clear();
            pvMat.clear();
            vMat.clear();
            pSigma.clear();

            for (int k = 0; k < this->m_k; k++) {
                S1 = Vec2Symmat(this->m_muMan.block(1, k, 3, 1));
                S2 = Vec2Symmat(xHat.col(t));
                Ac(0, 0) = 1;
                Ac.bottomRightCorner(2, 2) = ParallelTransport(S1[0], S2[0]);

                pvMatK.clear();
                vMatK.clear();
                pvKTmp.setZero();
                for (int j = 0; j < V[0].cols(); j++) {
                    vMatTmp.setZero();
                    vMatTmp(0, 0) = V[k](0, j);
                    vMatTmp.bottomRightCorner(2, 2) = Vec2Symmat(
                            V[k].block(1, j, 3, 1))[0]; //Vec2Symmat only one matrix here
                    vMatK.push_back(vMatTmp);

                    pvMatK.push_back(Ac * pow(D[k](j, j), 0.5) * vMatK[j] * Ac.transpose());
                    pvKTmp(0, j) = pvMatK[j](0, 0);
                    pvKTmp.col(j).bottomRows(3) = Symmat2Vec(pvMatK[j].block(1, 1, 2, 2)).transpose();
                }

                pvMat.push_back(pvMatK);
                vMat.push_back(vMatK);
                pv.push_back(pvKTmp);

                // Parallel transported sigma
                pSigma.push_back(pv[k] * pv[k].transpose());

                // Gaussian conditioning on tangent space
                uoutTmp.col(k) = LogmapVec(this->m_muMan.block(1, k, 3, 1), xHat.col(t))[0].transpose() +
                                 (getOutIn(pSigma[k]).array() / pSigma[k](0, 0) *
                                  (xIn(0, t) - this->m_muMan(0, k))).matrix();
                uHat.col(t) = uHat.col(t) + uoutTmp.col(k) * H(k, t);
            }
            xHat.col(t) = ExpmapVec(uHat.col(t), xHat.col(t))[0].transpose();
            cout << "Iteration done." << endl;
        }
        uOut.push_back(uoutTmp);
        expSigmaT.setZero();
        for (int k = 0; k < this->m_k; k++) {
            SigmaOutTmp = getOutOut(pSigma[k]) -
                          (getOutIn(pSigma[k]) / pSigma[k](0, 0)) * getInOut(pSigma[k]).transpose();

        expSigmaT = expSigmaT + H(k, t) * (SigmaOutTmp + uOut[t].col(k) * uOut[t].col(k).transpose());
        }
        expSigma.push_back(expSigmaT - uHat.col(t) * uHat.col(t).transpose());
    }

    for(int ttt=0;ttt<expSigma.size();ttt++){
        cout<<"\n"<<endl;
        cout<<expSigma[ttt]<<endl;
    }
}