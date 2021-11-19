#include "GMM_SPD.h"

#define deb(x) cout << #x << " " << x << endl;
#define dimensions 3

GMM_SPD::GMM_SPD() {
    this->m_k = 5; // TODO when 15 3dlearning fails in SPD EM, with 5 ok
    this->m_n = -1;
    this->m_maxDiffLL = 1e-4; //Likelihood increase threshold to stop algorithm
    this->m_minIterEM=1;
    this->m_maxIterEM = 10;
    this->m_maxIterM = 10;

    if(dimensions==2){
        this->m_dimVar = 4;
        this->m_dimIn = 1;
        this->m_dimOut = 3;
        this->m_dimOutVec = 2 + 2 * (2 - 1) / 2;
        this->m_dimVarVec = 3 - 2 + this->m_dimOutVec;
        this->m_dimCovOut = 3 + 3 * (3 - 1) / 2;
    }
    else{
        this->m_dimVar = 7;
        this->m_dimIn = 1;
        this->m_dimOut = 6;
        this->m_dimOutVec = 4;
//        this->m_dimOutVec = 3 + 2 * (3 - 1) / 2;
        this->m_dimVarVec = 6 - 3 + this->m_dimOutVec;
        this->m_dimCovOut = 4 + 2 * (4 - 1) / 2;
    }

//    this->m_dimOutVec = this->m_dimOut + this->m_dimOut * (this->m_dimOut - 1) / 2;
//    this->m_dimVarVec = this->m_dimVar - this->m_dimOut + this->m_dimOutVec;
//    this->m_dimCovOut = this->m_dimVar + this->m_dimVar * (this->m_dimVar - 1) / 2;

    this->m_dt = 1e-2;
    this->m_regTerm = 1e-4;
    this->m_kp = 100;
    this->m_km=10;
}

MatrixXd GMM_SPD::getInOut(const MatrixXd& m) {
    return m.block(0, this->m_dimIn, this->m_dimIn, this->m_dimOut);
}

MatrixXd GMM_SPD::getInIn(const MatrixXd& m) {
    return m.block(0, 0, this->m_dimIn, this->m_dimIn);
}

MatrixXd GMM_SPD::getOutIn(const MatrixXd& m) {
    return m.block(this->m_dimIn, 0, this->m_dimOut, this->m_dimIn);
}

MatrixXd GMM_SPD::getOutOut(const MatrixXd& m) {
    return m.block(this->m_dimIn, this->m_dimIn, this->m_dimOut, this->m_dimOut);
}

//Checked!
void GMM_SPD::InitModel(const MatrixXd& data, int demos) {
    this->m_nDemos = demos;  //4
    this->m_n = data.rows() / this->m_nDemos; //nbData
    this->m_nData = data.rows(); //nb
    if(dimensions==2) this->m_muMan = MatrixXd(4, this->m_k);
    else this->m_muMan = MatrixXd(7, this->m_k);
    this->m_mu = MatrixXd(this->m_muMan.rows(), this->m_muMan.cols());
    this->m_mu.setZero();
    this->m_data = data;
    this->m_muMan.setZero();

//    this->m_muMan << 0.105000000000000, 0.305000000000000, 0.505000000000000, 0.705000000000000, 0.905000000000000,
//            146.096605076556, 50.2725428695640, 57.5434325637446, 61.0162279884024, 48.6986348781330,
//            44.2213672844795, 70.3227237095515, 52.9208375483169, 43.8637203321041, 176.322381110539,
//            -71.9140893269118, 33.7271691014874, -43.1366974235137, 45.0107695266492, 106.855203790622;

    vector<int> timing = linspace(0, this->m_n, this->m_k + 1);
    vector<int> collected;

    for (int i = 0; i < this->m_k; i++) {
        collected.clear();
        for (int d = 0; d < this->m_nDemos; d++) {
            for (int t = timing[i]; t < timing[i + 1]; t++) {
                collected.push_back(d * this->m_n + t);
            }
        }

        this->m_priors.push_back(collected.size());
        MatrixXd collectedMatrix;
        MatrixXd collectedMatrixFull; // with time
        if(dimensions==2) {
            collectedMatrix = MatrixXd(collected.size(), 3);
            collectedMatrixFull = MatrixXd(collected.size(), 4);
        }
        else{
            collectedMatrix=MatrixXd(collected.size(), 6);
            collectedMatrixFull=MatrixXd(collected.size(), 7);
        }
        for (int l = 0; l < collected.size(); l++) {
            if(dimensions==2) {
                collectedMatrix.block(l, 0, 1, 3) = data.row(collected[l]).rightCols(3); // collected matrices checked!
            }
            else {
                collectedMatrix.block(l, 0, 1, 6) = data.row(collected[l]).rightCols(6); // collected matrices checked!
            }
            collectedMatrixFull.row(l) = data.row(collected[l]);
        }

        //TODO Find out why small numerical errors to MATLAB code here (in range 1e-03)

        // MuMan checked!
        this->m_muMan.col(i) = collectedMatrixFull.colwise().mean();
        if(dimensions==2) this->m_muMan.block(1,i,3,1) = symmat2Vec(
                    spdMean(vec2Symmat(collectedMatrix.transpose()), 10)).transpose();
        else this->m_muMan.block(1,i,6,1) = symmat2Vec(spdMean(vec2Symmat(collectedMatrix.transpose()), 10)).transpose();

        vector<MatrixXd> dataTangent;
        // DataTangent checked!
        if(dimensions==2) dataTangent = logMapVec(collectedMatrix.transpose(),
                                                  this->m_muMan.col(i).bottomRows(3)); // cut off t data
        else dataTangent = logMapVec(collectedMatrix.transpose(),
                                     this->m_muMan.col(i).bottomRows(6)); // cut off t data

        MatrixXd dataTangentMatrix;
        if(dimensions==2) dataTangentMatrix=MatrixXd(4, dataTangent.size());
        else dataTangentMatrix=MatrixXd(7, dataTangent.size());
        dataTangentMatrix.setZero();
        for (int i = 0; i < dataTangent.size(); i++) {
            dataTangentMatrix(0, i) = data(collected[i], 0);
            if(dimensions==2) dataTangentMatrix.block(1, i, 3, 1) = dataTangent[i].transpose(); //to matrix as in matlab code, with t row
            else dataTangentMatrix.block(1, i, 6, 1) = dataTangent[i].transpose(); //to matrix as in matlab code, with t row
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

    this->m_L = MatrixXd(this->m_k, data.rows());
    this->m_L.setZero();
    this->m_gamma = MatrixXd(this->m_k, data.cols());
    this->m_H = MatrixXd(this->m_k, data.cols());
}

//Checked! (numerically slightly different)
VectorXd GMM_SPD::GaussPDF(const MatrixXd& data, const MatrixXd& mu, const MatrixXd& sig) {
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
    pdf = tmp * dataCentered;
    pdf = exp(-0.5 * pdf) / sqrt(2 * M_PI * abs(sig) + numeric_limits<double>::min());
    return pdf;
}

// Checked -> small numerical errors
void GMM_SPD::EStep() {
    for (int k = 0; k < this->m_k; k++) {
        MatrixXd tmp(1, this->m_data.rows());
//        MatrixXd tmp(1, this->m_n * m_nDemos);
        tmp.setConstant(this->m_muMan(0, k));
        MatrixXd xts(this->m_dimVarVec, this->m_data.rows());
        xts.setZero();
        xts.row(0) = this->m_data.col(0).transpose() - tmp;

        vector<MatrixXd> logmapvec;
        if(dimensions==2) logmapvec = logMapVec(this->m_data.transpose().bottomRows(3),
                                                this->m_muMan.block(1, k, 3, 1)); //row vector
        else logmapvec = logMapVec(this->m_data.transpose().bottomRows(6),
                                   this->m_muMan.block(1, k, 6, 1)); //row vector
        for (int v = 0; v < logmapvec.size(); v++) {
            if(dimensions==2) xts.block(1, v, 3, 1) = logmapvec[v].transpose();
            else xts.block(1, v, 6, 1) = logmapvec[v].transpose();
        }
        this->m_xts.push_back(xts);
        this->m_L.row(k) =
                this->m_priors[k] * GaussPDF(this->m_xts[k], this->m_mu.col(k), this->m_sigma[k]).transpose();
    }
    this->m_gamma = (this->m_L.array() /
                     (this->m_L.colwise().sum().array() + numeric_limits<double>::min()).replicate(this->m_k,
                                                                                                   1).array()).matrix();
    this->m_H = (this->m_gamma.array() /
                 (this->m_gamma.rowwise().sum().array() + numeric_limits<double>::min()).replicate(1, this->m_n *
                                                                                                      this->m_nDemos).array()).matrix();
}

//Checked!
void GMM_SPD::MStep() {
    MatrixXd uTmp(this->m_dimVarVec, this->m_data.rows());
    MatrixXd tmp(1, this->m_n * m_nDemos);
    MatrixXd tmpId = MatrixXd::Identity(this->m_dimVarVec, this->m_dimVarVec);
    MatrixXd uTmpTot;
    vector<MatrixXd> logmapvec;
    for (int k = 0; k < this->m_k; k++) {
        this->m_priors[k] = this->m_gamma.row(k).sum() / this->m_data.rows();
        for (int n = 0; n < this->m_maxIterM; n++) {
            //Upd on tangent space
            uTmp.setZero();
            logmapvec.clear();
            tmp.setConstant(this->m_muMan(0, k));
            uTmp.row(0) = this->m_data.col(0).transpose() - tmp;
            if(dimensions==2) logmapvec = logMapVec(this->m_data.transpose().bottomRows(3),
                                                    this->m_muMan.block(1, k, 3, 1)); //row vector
            else logmapvec = logMapVec(this->m_data.transpose().bottomRows(6),
                                       this->m_muMan.block(1, k, 6, 1)); //row vector
            for (int v = 0; v < logmapvec.size(); v++) {
                if(dimensions==2) uTmp.block(1, v, 3, 1) = logmapvec[v].transpose();
                else uTmp.block(1, v, 6, 1) = logmapvec[v].transpose();
            }
            uTmpTot = (uTmp.array() *
                       (this->m_H.row(k).replicate(this->m_dimVarVec, 1)).array()).matrix().rowwise().sum();

            //Upd on manifold
            this->m_muMan(0, k) = uTmpTot(0) + this->m_muMan(0, k);
            if(dimensions==2) this->m_muMan.block(1, k, 3, 1) = expmapVec(uTmpTot.bottomRows(3),
                                                                          this->m_muMan.block(1, k, 3, 1))[0].transpose();
            else this->m_muMan.block(1, k, 6, 1) = expmapVec(uTmpTot.bottomRows(6),
                                                             this->m_muMan.block(1, k, 6, 1))[0].transpose();
        }
        this->m_sigma[k] = (uTmp * this->m_H.row(k).asDiagonal() * uTmp.transpose()) +
                           (tmpId * this->m_regTerm);
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
    std::vector<float> LL;
    for (int iter = 0; iter < this->m_maxIterEM; iter++) {
        cout << "EM iteration #" << iter << " ";
        EStep();
        cout << "e step done ";
        MStep();
        cout << "m step done" << endl;
        LL.push_back(this->m_L.colwise().sum().array().log().sum() / this->m_data.cols()); //daviates by 0.005 from MATLAB code
        cout<<LL[iter] - LL[iter - 1]<<endl;

        if(iter>=this->m_minIterEM && (LL[iter] - LL[iter - 1] < this->m_maxDiffLL || iter == this->m_maxIterEM - 1)){
            std::cout<< "Converged after "<<std::to_string(iter) <<" iterations. "<<std::endl;
            return;
        }
    }
    cout << " The maximum number of iterations has been reached." << endl;
}

void GMM_SPD::SigmaEigenDecomposition(const vector<MatrixXd>& Sigma, vector<MatrixXd>& V, vector<MatrixXd>& D) {
    for (int i = 0; i < Sigma.size(); i++) {
        EigenSolver<MatrixXd> es(Sigma[i]);
        D.push_back(es.eigenvalues().real().asDiagonal());
        V.push_back(es.eigenvectors().real());
    }
}

void GMM_SPD::GMR(MatrixXd& xHat, vector<MatrixXd>& sigmaXd) {
    MatrixXd xIn = VectorXd::LinSpaced(this->m_n, this->m_dt,
                                       this->m_n * this->m_dt).matrix().transpose(); // 1x100


    MatrixXd expSigmaT,uHat,Ac, uoutTmp, vMatTmp, pvKTmp;
    if(dimensions==2){
        expSigmaT=MatrixXd(3, 3);
        uHat=MatrixXd(3, this->m_n);
//        xHat=MatrixXd(3, this->m_n);
        Ac=MatrixXd(3, 3);
        uoutTmp=MatrixXd(3, this->m_k);
        vMatTmp=MatrixXd(3, 3);
        pvKTmp=MatrixXd(4, 4);

    }
    else{
        expSigmaT=MatrixXd(6, 6);
        uHat=MatrixXd(6, this->m_n);
//        xHat=MatrixXd(6, this->m_n);
        Ac=MatrixXd(6, 6);
        uoutTmp=MatrixXd(6, this->m_k);
        vMatTmp=MatrixXd(6, 6);
        pvKTmp=MatrixXd(7, 7);
    }

    vector<MatrixXd> S1, S2, uOut, expSigma, pvMatTmp, V, D, pvMatK, vMatK, pv, pSigma; //uout: 3x5x400, expsigma:3x3x400
    MatrixXd H(this->m_k, this->m_n);

    H.setZero();
    uHat.setZero();
    xHat.setZero();
    Ac.setZero();
    MatrixXf::Index max_index;
    MatrixXd SigmaOutTmp;

    SigmaEigenDecomposition(this->m_sigma, V, D);
    vector<vector<MatrixXd>> vMat, pvMat;

    cout<< "GMR ..."<<endl;

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
            if(dimensions==2) xHat.col(t) = this->m_muMan.block(1, max_index, 3, 1);
            else xHat.col(t) = this->m_muMan.block(1, max_index, 6, 1);
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
                if(dimensions==2){
                    S1 = vec2Symmat(this->m_muMan.block(1, k, 3, 1));
                    S2 = vec2Symmat(xHat.col(t));
                    Ac(0, 0) = 1;
                    Ac.bottomRightCorner(2, 2) = parallelTransport(S1[0], S2[0]);
                }
                else{
                    S1 = vec2Symmat(this->m_muMan.block(1, k, 6, 1));
                    S2 = vec2Symmat(xHat.col(t));
                    Ac(0, 0) = 1;
                    Ac.bottomRightCorner(3, 3) = parallelTransport(S1[0], S2[0]);
                }

                pvMatK.clear();
                vMatK.clear();
                pvKTmp.setZero();
                for (int j = 0; j < V[0].cols(); j++) {
                    if(dimensions==2){
                        vMatTmp.setZero();
                        vMatTmp(0, 0) = V[k](0, j);
                        vMatTmp.bottomRightCorner(2, 2) = vec2Symmat(
                                V[k].block(1, j, 3, 1))[0]; //vec2Symmat only one matrix here
                        vMatK.push_back(vMatTmp);

                        pvMatK.push_back(Ac * pow(D[k](j, j), 0.5) * vMatK[j] * Ac.transpose());
                        pvKTmp(0, j) = pvMatK[j](0, 0);
                        pvKTmp.col(j).bottomRows(3) = symmat2Vec(pvMatK[j].block(1, 1, 2, 2)).transpose();
                    }
                    else{
                        vMatTmp.setZero();
                        vMatTmp(0, 0) = V[k](0, j);
                        vMatTmp.bottomRightCorner(3, 3) = vec2Symmat(
                                V[k].block(1, j, 6, 1))[0]; //vec2Symmat only one matrix here
                        vMatK.push_back(vMatTmp);

                        pvMatK.push_back(Ac * pow(D[k](j, j), 0.5) * vMatK[j] * Ac.transpose());
                        pvKTmp(0, j) = pvMatK[j](0, 0);
                        pvKTmp.col(j).bottomRows(6) = symmat2Vec(pvMatK[j].block(1, 1, 3, 3)).transpose();
                    }
                }

                pvMat.push_back(pvMatK);
                vMat.push_back(vMatK);
                pv.push_back(pvKTmp);

                // Parallel transported sigma
                pSigma.push_back(pv[k] * pv[k].transpose());

                // Gaussian conditioning on tangent space
                if(dimensions==2) uoutTmp.col(k) = logMapVec(this->m_muMan.block(1, k, 3, 1), xHat.col(t))[0].transpose() +
                                                   (getOutIn(pSigma[k]).array() / pSigma[k](0, 0) *
                                  (xIn(0, t) - this->m_muMan(0, k))).matrix();
                else uoutTmp.col(k) = logMapVec(this->m_muMan.block(1, k, 6, 1), xHat.col(t))[0].transpose() +
                                      (getOutIn(pSigma[k]).array() / pSigma[k](0, 0) *
                                                    (xIn(0, t) - this->m_muMan(0, k))).matrix();
                uHat.col(t) = uHat.col(t) + uoutTmp.col(k) * H(k, t);
            }
            xHat.col(t) = expmapVec(uHat.col(t), xHat.col(t))[0].transpose();
        }
        uOut.push_back(uoutTmp);
        expSigmaT.setZero();
        for (int k = 0; k < this->m_k; k++) {
            SigmaOutTmp = getOutOut(pSigma[k]) -
                          (getOutIn(pSigma[k]) / pSigma[k](0, 0)) * getOutIn(pSigma[k]).transpose();

            expSigmaT = expSigmaT + H(k, t) * (SigmaOutTmp + uOut[t].col(k) * uOut[t].col(k).transpose());
        }
//        xd.col(t)=xHat.col(t);
        sigmaXd.push_back(expSigmaT - uHat.col(t) * uHat.col(t).transpose());
    }
}

void GMM_SPD::GMR(MatrixXd& xHat, vector<MatrixXd>& sigmaXd, int t) {
    MatrixXd xIn = VectorXd::LinSpaced(this->m_n, this->m_dt,
                                       this->m_n * this->m_dt).matrix().transpose(); // 1x100


    MatrixXd expSigmaT,uHat,Ac, uoutTmp, vMatTmp, pvKTmp;
    if(dimensions==2){
        expSigmaT=MatrixXd(3, 3);
        uHat=MatrixXd(3, this->m_n);
//        xHat=MatrixXd(3, this->m_n);
        Ac=MatrixXd(3, 3);
        uoutTmp=MatrixXd(3, this->m_k);
        vMatTmp=MatrixXd(3, 3);
        pvKTmp=MatrixXd(4, 4);

    }
    else{
        expSigmaT=MatrixXd(6, 6);
        uHat=MatrixXd(6, this->m_n);
//        xHat=MatrixXd(6, this->m_n);
        Ac=MatrixXd(6, 6);
        uoutTmp=MatrixXd(6, this->m_k);
        vMatTmp=MatrixXd(6, 6);
        pvKTmp=MatrixXd(7, 7);
    }

    vector<MatrixXd> S1, S2, uOut, expSigma, pvMatTmp, V, D, pvMatK, vMatK, pv, pSigma; //uout: 3x5x400, expsigma:3x3x400
    MatrixXd H(this->m_k, this->m_n);

    H.setZero();
    uHat.setZero();
    xHat.setZero();
    Ac.setZero();
    MatrixXf::Index max_index;
    MatrixXd SigmaOutTmp;

    SigmaEigenDecomposition(this->m_sigma, V, D);
    vector<vector<MatrixXd>> vMat, pvMat;

        for (int k = 0; k < this->m_k; k++) {
            H(k, t) = this->m_priors[k] *
                      GaussPDF(xIn(0, t) - this->m_muMan(0, k), this->m_mu(0, k), this->m_sigma[k](0, 0));
        }
        H.col(t) = (H.col(t).array() / (H.col(t).array() + numeric_limits<double>::min()).colwise().sum()(0,
                                                                                                          0)).matrix(); // sum only one value here

        // Compute conditional mean (with covariance transportation)
        if (t == 0) {
            H.col(t).maxCoeff(&max_index);
            if(dimensions==2) xHat.col(t) = this->m_muMan.block(1, max_index, 3, 1);
            else xHat.col(t) = this->m_muMan.block(1, max_index, 6, 1);
        } else {
            xHat.col(t) = xHat.col(t - 1);
        }

        // Iterative computation
        for (int iter = 0; iter < 10; iter++) {
            deb("In SPD_GMR iter")
            deb(iter)
            uHat.col(t).setZero();
            uoutTmp.setZero();

            pv.clear();
            pvMat.clear();
            vMat.clear();
            pSigma.clear();

            for (int k = 0; k < this->m_k; k++) {
                if(dimensions==2){
                    S1 = vec2Symmat(this->m_muMan.block(1, k, 3, 1));
                    S2 = vec2Symmat(xHat.col(t));
                    Ac(0, 0) = 1;
                    Ac.bottomRightCorner(2, 2) = parallelTransport(S1[0], S2[0]);
                }
                else{
                    S1 = vec2Symmat(this->m_muMan.block(1, k, 6, 1));
                    S2 = vec2Symmat(xHat.col(t));
                    Ac(0, 0) = 1;
                    Ac.bottomRightCorner(3, 3) = parallelTransport(S1[0], S2[0]);
                }

                pvMatK.clear();
                vMatK.clear();
                pvKTmp.setZero();
                for (int j = 0; j < V[0].cols(); j++) {
                    if(dimensions==2){
                        vMatTmp.setZero();
                        vMatTmp(0, 0) = V[k](0, j);
                        vMatTmp.bottomRightCorner(2, 2) = vec2Symmat(
                                V[k].block(1, j, 3, 1))[0]; //vec2Symmat only one matrix here
                        vMatK.push_back(vMatTmp);

                        pvMatK.push_back(Ac * pow(D[k](j, j), 0.5) * vMatK[j] * Ac.transpose());
                        pvKTmp(0, j) = pvMatK[j](0, 0);
                        pvKTmp.col(j).bottomRows(3) = symmat2Vec(pvMatK[j].block(1, 1, 2, 2)).transpose();
                    }
                    else{
                        vMatTmp.setZero();
                        vMatTmp(0, 0) = V[k](0, j);
                        vMatTmp.bottomRightCorner(3, 3) = vec2Symmat(
                                V[k].block(1, j, 6, 1))[0]; //vec2Symmat only one matrix here
                        vMatK.push_back(vMatTmp);

                        pvMatK.push_back(Ac * pow(D[k](j, j), 0.5) * vMatK[j] * Ac.transpose());
                        pvKTmp(0, j) = pvMatK[j](0, 0);
                        pvKTmp.col(j).bottomRows(6) = symmat2Vec(pvMatK[j].block(1, 1, 3, 3)).transpose();
                    }
                }

                pvMat.push_back(pvMatK);
                vMat.push_back(vMatK);
                pv.push_back(pvKTmp);

                // Parallel transported sigma
                pSigma.push_back(pv[k] * pv[k].transpose());

                // Gaussian conditioning on tangent space
                if(dimensions==2) uoutTmp.col(k) = logMapVec(this->m_muMan.block(1, k, 3, 1), xHat.col(t))[0].transpose() +
                                                   (getOutIn(pSigma[k]).array() / pSigma[k](0, 0) *
                                                    (xIn(0, t) - this->m_muMan(0, k))).matrix();
                else uoutTmp.col(k) = logMapVec(this->m_muMan.block(1, k, 6, 1), xHat.col(t))[0].transpose() +
                                      (getOutIn(pSigma[k]).array() / pSigma[k](0, 0) *
                                       (xIn(0, t) - this->m_muMan(0, k))).matrix();
                uHat.col(t) = uHat.col(t) + uoutTmp.col(k) * H(k, t);
            }
            xHat.col(t) = expmapVec(uHat.col(t), xHat.col(t))[0].transpose();
        }
        uOut.push_back(uoutTmp);
        expSigmaT.setZero();
        for (int k = 0; k < this->m_k; k++) {
            SigmaOutTmp = getOutOut(pSigma[k]) -
                          (getOutIn(pSigma[k]) / pSigma[k](0, 0)) * getOutIn(pSigma[k]).transpose();

            expSigmaT = expSigmaT + H(k, t) * (SigmaOutTmp + uOut[t].col(k) * uOut[t].col(k).transpose());
        }
//        xd.col(t)=xHat.col(t);
        sigmaXd.push_back(expSigmaT - uHat.col(t) * uHat.col(t).transpose());
}
