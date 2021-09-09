#include "GMM_SPD.h"
#include <Eigen/Core>

GMM_SPD::GMM_SPD() {
    this->m_k = 5;
//    this->m_nks=std::vector<double>(this->m_k, 0.0);
    this->m_n = 0;
    this->m_maxDiffLL=1e-4; //Likelihood increase threshold to sop algorithm
    this->m_maxIterEM=1;
    this->m_maxIterM=10;
    this->m_dimOut=2;
    this->m_dimOutVec = this->m_dimOut + this->m_dimOut * (this->m_dimOut - 1) / 2;
    this->m_dimVar = 3;
    this->m_dimVarVec = this->m_dimVar - this->m_dimOut + this->m_dimOutVec;
    this->m_dimCovOut = this->m_dimVar + this->m_dimVar * (this->m_dimVar - 1) / 2;
    this->m_dt = 1e-2;
    this->m_regTerm = 1e-4;
    this->m_kp=100;
    this->m_nDemos=4;
}

// Checked!
std::vector<int> GMM_SPD::linspace(double a, double b, std::size_t N)
{
    double h = (b - a) / static_cast<double>(N-1);
    std::vector<int> xs(N);
    std::vector<int>::iterator x;
    double val;
    for (x = xs.begin(), val = a; x != xs.end(); ++x, val += h) {
        *x =(int) round(val);
    }
    return xs;
}

// Checked!
void GMM_SPD::CumulativeSum(const Eigen::VectorXd& input, Eigen::VectorXd& result) {
    result(0)=input[0];
    for (int i = 1; i < input.size(); i++) {
        result(i) = result(i - 1) + input(i);
    }
}

//Checked!
void GMM_SPD::InitModel(Eigen::MatrixXd *data){
    this->m_n = data->rows() / this->m_nDemos; //nbData
    this->m_nData = data->rows() ; //nb
    this->m_muMan= MatrixXd(4, this->m_k);
    this->m_mu= MatrixXd(this->m_muMan.rows(), this->m_muMan.cols());
    this->m_mu.setZero();
    this->m_data = *data;

//    std::cout<<"\nTHIS DATA\n"<<std::endl;
//    std::cout<<this->m_data<<std::endl;

//    this->m_muMan=MatrixXd(4,this->m_k);
    this->m_muMan.setZero();
//    this->m_muMan <<0.105000000000000,	0.305000000000000,	0.505000000000000,	0.705000000000000,	0.905000000000000,
//            146.096605076556,	50.2725428695640,	57.5434325637446,	61.0162279884024,	48.6986348781330,
//            44.2213672844795,	70.3227237095515,	52.9208375483169,	43.8637203321041,	176.322381110539,
//            -71.9140893269118,	33.7271691014874,	-43.1366974235137,	45.0107695266492,	106.855203790622;

    std::vector<int> timing = linspace(0, this->m_n, this->m_k+1);
    for(int i=0; i<this->m_k;i++){
        std::vector<int> collected;
        for(int d=0;d<this->m_nDemos; d++) {
            for(int t=timing[i];t<timing[i+1]; t++) {
            collected.push_back(d * this->m_n + t);
            }
        }
        this->m_priors.push_back(collected.size());
        MatrixXd collectedMatrix(collected.size(),3);
        MatrixXd collectedMatrixFull(collected.size(),4); // with time
        for(int l=0; l<collected.size();l++){
            collectedMatrix.block(l,0,1,3) = (*data).row(collected[l]).rightCols(3); // collected matrices checked!
            collectedMatrixFull.row(l) = (*data).row(collected[l]);
        }

        //TODO Find out why small numerical errors to MATLAB code here (in range 1e-03)

        // MuMan checked!
        this->m_muMan.col(i) = collectedMatrixFull.colwise().mean();
        this->m_muMan.block(1,i,3,1) = Symmat2Vec(SPDMean(Vec2Symmat(collectedMatrix.transpose()), 10)).transpose();

        // DataTangent checked!
        std::vector<MatrixXd> dataTangent = LogmapVec(collectedMatrix.transpose(), this->m_muMan.col(i).bottomRows(3)); // cut off t data
        MatrixXd dataTangentMatrix(4, dataTangent.size());
        dataTangentMatrix.setZero();
        for(int i=0;i<dataTangent.size();i++){
            dataTangentMatrix(0,i) = (*data)(collected[i],0);
            dataTangentMatrix.block(1,i,3,1) = dataTangent[i].transpose(); //to matrix as in matlab code, with t row
        }

        // Cov calculation checked!
        MatrixXd centeredTangent = dataTangentMatrix.colwise() - dataTangentMatrix.rowwise().mean();
//        MatrixXd cov = (centeredTangent * centeredTangent.adjoint()) / double(dataTangentMatrix.cols() - 1);
        MatrixXd cov = (centeredTangent * centeredTangent.adjoint()) / double(dataTangentMatrix.cols() - 1)+ MatrixXd(this->m_dimVarVec, this->m_dimVarVec).setConstant(this->m_regTerm);
        this->m_sigma.push_back(cov);
    }

    // Priors checked!
    double priorsSum = (double) (std::accumulate(this->m_priors.begin(), this->m_priors.end(), 0.0f));
    for(double& d : this->m_priors){
        d /= priorsSum;
    }

    this->m_L = MatrixXd(this->m_k, data->rows());
    this->m_L.setZero();
    this->m_gamma = MatrixXd(this->m_k, data->cols());
    this->m_H = MatrixXd(this->m_k, data->cols());

    std::cout << "\n\n\nAFTER INITIALISATION\n" << std::endl;
    std::cout << "muMan= \n" << this->m_muMan << std::endl; // error of up to 0.3
    for(int k=0; k<this->m_k; k++) {
        std::cout << "sigma= \n" << this->m_sigma[k] << std::endl;
        std::cout << "prior= \n" << this->m_priors[k] << std::endl;
    }
}


// Checked!
Eigen::MatrixXd GMM_SPD::SPDMean(std::vector<Eigen::MatrixXd> mat, int nIter) {
    MatrixXd M = mat[0];
    for(int iter=0;iter<nIter;iter++){
        MatrixXd L(mat[0].rows(), mat[0].cols());
        L.setZero();
        for(int i=0;i<mat.size();i++){
            MatrixXd tmp = M.pow(-0.5)*mat[i]*(M.pow(-0.5));
            L=L + (tmp.log()).matrix();
        }
        M=M.pow(0.5)*(L.array() / mat.size()).matrix().exp().matrix()*M.pow(0.5);
    }
    return M;
}

// Checked!
Eigen::MatrixXd GMM_SPD::Symmat2Vec(Eigen::MatrixXd mat) {
    int N=mat.rows();
    std::vector<double> v;
    Eigen::VectorXd dia = mat.diagonal();
    for(int x=0;x<dia.size();x++){
        v.push_back(dia(x));
    }
    int  row, col;
    for(int n=1;n<N;n++){
        row=0;
        col=n;
        for(int ni=n;ni<N;ni++) {
            v.push_back(sqrt(2) * mat(row, col));
            row++;
            col++;
        }
    }
    MatrixXd vm(1,v.size());
    for(int x=0;x<v.size();x++){
        vm(0,x)=v[x]; //one row
    }
    return vm;
}

// Checked!
std::vector<Eigen::MatrixXd> GMM_SPD::Symmat2Vec(std::vector<Eigen::MatrixXd> mat_vec) {
    int N =mat_vec.size();
    std::vector<Eigen::MatrixXd> vec;
    for(int i=0; i<N;i++){
        Eigen::MatrixXd vn = Symmat2Vec(mat_vec[i]);
        vec.push_back(vn);
    }
    return vec;
}

// Checked!
std::vector<Eigen::MatrixXd> GMM_SPD::Vec2Symmat(Eigen::MatrixXd vec) {
    std::vector<Eigen::MatrixXd> MVector;
    int d= vec.rows();
    int N = vec.cols();
    int D = (-1+ sqrt(1+8*d))/(double) 2;
    int row;
    for(int n=0;n<N;n++){ //colwise
        Eigen::MatrixXd vn = vec.col(n).transpose();
        Eigen::MatrixXd Mn= vn.row(0).leftCols(D).asDiagonal();
        Eigen::VectorXd id(D);
        CumulativeSum(Eigen::VectorXd::LinSpaced( D,  D, 1), id);
        Eigen::MatrixXd tmp1(Mn.rows(), Mn.cols());
        Eigen::MatrixXd tmp2(Mn.rows(), Mn.cols());
        for(int i=1;i<D;i++){
            tmp1.setZero();
            row=0;
            for(int k=i;k<id(i)-id(i-1)+i;k++){
                tmp1(row,k) = vn(0,id(i-1)+row)* (1/sqrt(2));
                row++;
            }
            tmp2.setZero();
            row=0;
            for(int k=i;k<id(i)-id(i-1)+i;k++){
                tmp2(k,row) = vn(0,id(i-1)+row)* (1/sqrt(2));
                row++;
            }
            Mn=Mn+tmp1+tmp2;
        }
        MVector.push_back(Mn);
    }
    return MVector;
}

// Checked!
std::vector<Eigen::MatrixXd> GMM_SPD::Vec2Symmat(std::vector<Eigen::MatrixXd> vec) {
    MatrixXd v(vec[0].rows(), vec.size());
    for(int i=0; i< vec.size();i++){
        v.col(i) = vec[i];
    }
    return Vec2Symmat(v);
}

// Checked!
std::vector<MatrixXd> GMM_SPD::LogmapVec(MatrixXd x, MatrixXd s){
    std::vector<MatrixXd> X = Vec2Symmat(x);
//    std::cout<<"\n\n\nIN LOGMAP VEC (vec2symmat)\n"<<std::endl;
//    std::cout<<"x: \n"<<x<<std::endl;
//    std::cout<<"X: \n"<<X[0]<<std::endl;
//    std::cout<<"\n\n\nEND LOGMAP VEC\n"<<std::endl;
    std::vector<MatrixXd> S = Vec2Symmat(s);
    std::vector<MatrixXd> U = LogMap(X,S[0]); //Vec2Symmat gives back vector of size 1 here
    std::vector<MatrixXd> u= Symmat2Vec(U);
    return u;
}

// Checked!
std::vector<MatrixXd> GMM_SPD::LogMap(std::vector<MatrixXd> X, MatrixXd S){
    std::vector<MatrixXd> U;
    for(int i=0;i<X.size();i++){
        Eigen::MatrixXd tmp = (S.inverse())*X[i]; //A\B in MATLAB is a^-1 * B
        Eigen::EigenSolver<MatrixXd> es(tmp);
        MatrixXd D = es.eigenvalues().real().asDiagonal();
        MatrixXd V = es.eigenvectors().real();
//        std::cout<<"tmp:\n";
//        std::cout<<tmp<<std::endl;
//        std::cout<<"D:\n";
//        std::cout<<D<<std::endl;
//        std::cout<<"V:\n";
//        std::cout<<V<<std::endl;
//        std::cout<<"\nres:\n";
//        std::cout<<(D.diagonal());
//        std::cout<<"\nres2:\n";
        MatrixXd tmp2 = D.diagonal().array().log().matrix().asDiagonal().toDenseMatrix();
//        std::cout<<tmp2;
//        std::cout<<"\nres3:\n";
//        std::cout<<S*V*tmp2*V.inverse()<<std::endl;
        U.push_back(S*V*tmp2*V.inverse());
    }
    return U;
}

//Checked! (numerically slightly different)
VectorXd GMM_SPD::GaussPDF(MatrixXd data, MatrixXd mu, MatrixXd sig){
    Eigen::MatrixXd pdf(1, this->m_n);
    Eigen::MatrixXd dataCentered = data.transpose() - mu.transpose().replicate(data.cols(),1);
    Eigen::MatrixXd tmp = dataCentered*(sig.inverse());
    pdf = (tmp.array() * dataCentered.array()).matrix().rowwise().sum();
    pdf = (-0.5*pdf).array().exp() / sqrt(pow(2*M_PI, this->m_dimVar)*abs(sig.determinant())+std::numeric_limits<double>::min());
    return pdf;
}

// Checked -> small numerical errors
void GMM_SPD::EStep() {
    for(int k=0; k< this->m_k; k++){
        MatrixXd tmp(1,this->m_n * m_nDemos);
        tmp.setConstant(this->m_muMan(0,k));
        MatrixXd xts(this->m_dimVarVec, this->m_data.rows());
        xts.setZero();
        xts.row(0) = this->m_data.col(0).transpose()-tmp;

        std::vector<MatrixXd> logmapvec = LogmapVec(this->m_data.transpose().bottomRows(3), this->m_muMan.block(1,k,3,1)); //row vector
        for(int v=0;v<logmapvec.size();v++){
            xts.block(1,v,3,1) = logmapvec[v].transpose();
        }
        this->m_xts.push_back(xts);
//        std::cout<<"prior: \n"<<this->m_priors[k]<<std::endl;
        std::cout<<"HERE"<<std::endl;
        std::cout<<"this->m_xts[k].transpose(): \n"<<this->m_xts[k].transpose()<<std::endl; // already here instabilities
        std::cout<<"this->m_mu: \n"<<this->m_mu.col(k)<<std::endl;
        std::cout<<"this->m_sigma[k]: \n"<<this->m_sigma[k]<<std::endl;
        std::cout<<"GaussPDF(this->m_xts[k], this->m_mu.col(k), this->m_sigma[k]): \n"<<GaussPDF(this->m_xts[k], this->m_mu.col(k), this->m_sigma[k])<<std::endl;
        this->m_L.row(k) = this->m_priors[k] * GaussPDF(this->m_xts[k], this->m_mu.col(k), this->m_sigma[k]).transpose();
    }
//    this->m_gamma = (this->m_L.array() / (this->m_L.colwise().sum().array()).replicate(this->m_k, 1).array()).matrix();
    this->m_gamma = (this->m_L.array() / (this->m_L.colwise().sum().array()+std::numeric_limits<double>::min()).replicate(this->m_k, 1).array()).matrix();
//    this->m_H = (this->m_gamma.array() / (this->m_gamma.rowwise().sum().array()).replicate(1,this->m_n*this->m_nDemos).array()).matrix();
    this->m_H = (this->m_gamma.array() / (this->m_gamma.rowwise().sum().array()+std::numeric_limits<double>::min()).replicate(1,this->m_n*this->m_nDemos).array()).matrix();

}

//Checked!
void GMM_SPD::MStep() {
    for(int k=0; k<this->m_k; k++){
        this->m_priors[k] = this->m_gamma.row(k).sum() /this->m_data.rows();
        MatrixXd uTmp(this->m_dimVarVec, this->m_data.rows());
        for(int n=0; n<this->m_maxIterM; n++){

            //Upd on tangent space
            uTmp.setZero();
            MatrixXd tmp(1,this->m_n * m_nDemos);
            tmp.setConstant(this->m_muMan(0,k));
            uTmp.row(0) = this->m_data.col(0).transpose()-tmp;
            std::vector<MatrixXd> logmapvec = LogmapVec(this->m_data.transpose().bottomRows(3), this->m_muMan.block(1,k,3,1)); //row vector
            for(int v=0;v<logmapvec.size();v++){
                uTmp.block(1,v,3,1) = logmapvec[v].transpose();
            }
            MatrixXd uTmpTot = (uTmp.array() * (this->m_H.row(k).replicate(this->m_dimVarVec,1)).array()).matrix().rowwise().sum();

            //Upd on manifold
            this->m_muMan(0,k) = uTmpTot(0) + this->m_muMan(0,k);
            this->m_muMan.block(1,k,3,1) = ExpmapVec(uTmpTot.bottomRows(3), this->m_muMan.block(1,k,3,1))[0].transpose();
        }
        this->m_sigma[k] = (uTmp * this->m_H.row(k).asDiagonal() * uTmp.transpose()) + (MatrixXd(this->m_dimVarVec, this->m_dimVarVec).setIdentity()*this->m_regTerm);
    }

    std::cout << "\n\n\nAFTER EM ALGORITHM\n" << std::endl;
    std::cout << "muMan= \n" << this->m_muMan << std::endl;
    // If niter 10 and niterem 1 error same, if niter 1 and niterem 10 then absolutely high -> something happens in between? TODO check!
    // error up to 0.3, when using exact initialized muman, then only up to 0.02 (1,1)
    // when using exact initialized muman, up to 0.03 (10,1)
    // when using exact initialized muman, HUGE! (10,10)
    for(int k=0; k<this->m_k; k++) {
        std::cout << "sigma= \n" << this->m_sigma[k] << std::endl;
        std::cout << "prior= \n" << this->m_priors[k] << std::endl;
    }
}

void GMM_SPD::TrainEM(){
    for(int iter=0;iter< this->m_maxIterEM; iter++){
        std::cout<<"EM iteration #"<<iter<<" ";
        EStep();
        std::cout<<"e step done ";
        MStep();
        std::cout<<"m step done"<<std::endl;
    }
    std::cout<<" The maximum number of iterations has been reached."<<std::endl;
}

// Checked!
std::vector<MatrixXd> GMM_SPD::ExpmapVec(MatrixXd u, MatrixXd s) {
    std::vector<MatrixXd> U = Vec2Symmat(u);
    std::vector<MatrixXd> S = Vec2Symmat(s);
    std::vector<MatrixXd> X = ExpMap(U,S[0]); //Vec2Symmat gives back vector of size 1 here
    std::vector<MatrixXd> x= Symmat2Vec(X);
    return x;
}

// Checked!
std::vector<MatrixXd> GMM_SPD::ExpMap(std::vector<MatrixXd> U, MatrixXd S) {
    std::vector<MatrixXd> X;
    for(int i=0;i<U.size();i++){
        Eigen::MatrixXd tmp = (S.inverse())*U[i]; //A\B in MATLAB is a^-1 * B
        Eigen::EigenSolver<MatrixXd> es(tmp);
        MatrixXd D = es.eigenvalues().real().asDiagonal();
        MatrixXd V = es.eigenvectors().real();
        MatrixXd tmp2 = D.diagonal().array().exp().matrix().asDiagonal().toDenseMatrix();
        X.push_back(S*V*tmp2*V.inverse());
    }
    return X;
}

void GMM_SPD::SigmaEigenDecomposition(std::vector<MatrixXd> *Sigma, std::vector<MatrixXd> *V, std::vector<MatrixXd> *D) {
    for (int i = 0; i < Sigma->size(); i++) {
        Eigen::EigenSolver<MatrixXd> es((*Sigma)[i]);
        D->push_back(es.eigenvalues().real().asDiagonal());
        V->push_back(es.eigenvectors().real());
//        std::cout<<"tmp:\n";
//        std::cout<<tmp<<std::endl;
//        std::cout<<"D:\n";
//        std::cout<<D<<std::endl;
//        std::cout<<"V:\n";
//        std::cout<<V<<std::endl;
//        std::cout<<"\nres:\n";
//        std::cout<<(D.diagonal());
//        std::cout<<"\nres2:\n";

    }
}

void GMM_SPD::GMR(){
//    MatrixXd xIn= Eigen::VectorXd::LinSpaced(this->m_nData, 0, this->m_nData*this->m_dt).matrix();
//    MatrixXd uHat(3,this->m_nData);
//    MatrixXd xHat(3,this->m_nData);
//    std::vector<MatrixXd> uOut; //3x5x400
//    std::vector<MatrixXd> uexpSigma; //3x3x400
//    std::vector<MatrixXd> H;
//    uHat.setZero();
//    xHat.setZero();
    for(int t=0; t<this->m_nData;t++){

    }



}
