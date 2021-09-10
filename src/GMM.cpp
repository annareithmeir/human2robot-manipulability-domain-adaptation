#include "GMM.h"

GMM::GMM() {
    this->m_k = 5;
//    this->m_nks=std::vector<double>(this->m_k, 0.0);
    this->m_n = 0;
    this->m_maxDiffLL=1e-4; //Likelihood increase threshold to sop algorithm
    this->m_minIterEM=5;
    this->m_maxIterEM=100;
    this->m_dimOut=2;
    this->m_dimOutVec = this->m_dimOut + this->m_dimOut * (this->m_dimOut - 1) / 2;
    this->m_dimVar = 3;
    this->m_dimVarVec = this->m_dimVar - this->m_dimOut + this->m_dimOutVec;
    this->m_dimCovOut = this->m_dimVar + this->m_dimVar * (this->m_dimVar - 1) / 2;
    this->m_dt = 1e-2;
    this->m_regTerm = 1e-4;
    this->m_kp=100;
    this->m_nDemos=4;
    this->m_regTerm = 1e-4;
}

std::vector<double> GMM::linspace(double a, double b, std::size_t N)
{
    double h = (b - a) / static_cast<double>(N-1);
    std::vector<double> xs(N);
    std::vector<double>::iterator x;
    double val;
    for (x = xs.begin(), val = a; x != xs.end(); ++x, val += h) {
        *x = val;
    }
    return xs;
}

//Checked!
void GMM::InitModel(MatrixXd *data){
    this->m_n = data->cols() / this->m_nDemos;
    this->m_nData = data->cols();
    this->m_mu= MatrixXd(this->m_dimVar, this->m_k);
    this->m_data_pos = *data;

    std::vector<double> timing = linspace((*data)(0, 0), (*data)(0, data->cols() - 1), this->m_k + 1);
    for(int i=0; i<this->m_k;i++){
        int tmp=0;
        Eigen::MatrixXd collected(3, data->cols());
        collected.setZero();
        for(int t=0; t < data->cols(); t++) {
            if((*data)(0, t) >= timing[i] & (*data)(0, t) < timing[i + 1]) {
                collected(0,tmp) = (*data)(0, t);
                collected(1,tmp) = (*data)(1, t);
                collected(2,tmp) = (*data)(2, t);
//                collected(3,tmp) = (*data)(3,t);
                tmp++;
            }
        }

        collected = collected.leftCols(tmp);
        this->m_priors.push_back(tmp);
        this->m_mu.block(0, i, 3, 1)= collected.rowwise().mean();
        MatrixXd centered = collected.colwise() - collected.rowwise().mean();
        MatrixXd cov = (centered * centered.adjoint()) / double(collected.cols() - 1);
//        MatrixXd cov = (centered * centered.adjoint()) / double(collected.cols() - 1)+ MatrixXd(this->m_dimVar, this->m_dimVar).setConstant(this->m_regTerm);;
        this->m_sigma.push_back(cov);
    }
    double priorsSum = (double) (std::accumulate(this->m_priors.begin(), this->m_priors.end(), 0.0f));
    for(double& d : this->m_priors){
        d /= priorsSum;
    }

    this->m_L = MatrixXd(this->m_k, data->cols());
    this->m_gamma = MatrixXd(this->m_k, data->cols());
    this->m_gamma2 = MatrixXd(this->m_k, data->cols());
}

//Checked! (numerical slightly different
Eigen::VectorXd GMM::GaussPDF(Eigen::MatrixXd mu, Eigen::MatrixXd sig){
    Eigen::MatrixXd pdf(1, this->m_n);
    Eigen::MatrixXd dataCentered = this->m_data_pos.transpose() - mu.transpose().replicate(this->m_data_pos.cols(),1);
    Eigen::MatrixXd tmp = dataCentered*(sig.inverse());
    pdf = (tmp.array() * dataCentered.array()).matrix().rowwise().sum();
    pdf = (-0.5*pdf).array().exp() / sqrt(pow(2*M_PI, this->m_dimVar)*abs(sig.determinant())+std::numeric_limits<double>::min());
    return pdf;
}

double GMM::GaussPDF(double data, double mu, double sig){
    double pdf;
    double dataCentered = data- mu;
    double tmp = dataCentered/sig;
//    std::cout<<"\n\n IN GAUSS\n"<<std::endl;
//    std::cout<<"\ndata\n"<<data<<std::endl;
//    std::cout<<"\nmu\n"<<mu<<std::endl;
//    std::cout<<"\nsig\n"<<sig<<std::endl;
//
//    std::cout<<"\ndatacentered\n"<<dataCentered<<std::endl;
//    std::cout<<"\ntmp\n"<<tmp<<std::endl;
    pdf = tmp*dataCentered;
//    std::cout<<"\npdf\n"<<pdf<<std::endl;
    pdf = exp(-0.5*pdf) / sqrt(2*M_PI*abs(sig)+std::numeric_limits<double>::min());
//    std::cout<<"\npdf\n"<<pdf<<std::endl;
    return pdf;
}

void GMM::EStep() {
    for(int k=0; k< this->m_k; k++){
        this->m_L.row(k) = this->m_priors[k] * GaussPDF(this->m_mu.col(k), this->m_sigma[k]).transpose();
    }
    this->m_gamma = (this->m_L.array() / (this->m_L.colwise().sum().array()+std::numeric_limits<double>::min()).replicate(this->m_k, 1).array()).matrix();
}

//Checked!
void GMM::MStep() {
    Eigen::Vector3i updateComp; //m_priors, m_mu, m_sigma update
    updateComp.setOnes();

    for(int k=0; k<this->m_k; k++){
        if(updateComp(0) == 1){ //priors
            this->m_priors[k] = this->m_gamma.row(k).sum() /this->m_data_pos.cols();
        }
        if(updateComp(1) == 1){ //mu
            this->m_mu.col(k) = this->m_data_pos * this->m_gamma2.row(k).transpose();
        }
        if(updateComp(2) == 1){ //sigma
            Eigen::MatrixXd dataTmp = (this->m_data_pos.array() - this->m_mu.col(k).replicate(1,this->m_data_pos.cols()).array()).matrix();
            Eigen::MatrixXd tmp;
            this->m_sigma[k] =dataTmp * this->m_gamma2.row(k).asDiagonal() * dataTmp.transpose() + tmp.setIdentity(this->m_dimVar,this->m_dimVar)*this->m_regTerm;
        }
    }
}

void GMM::TrainEM(){
    std::vector<float> LL;
    this->m_maxIterEM=100;

    for(int iter=0;iter< this->m_maxIterEM; iter++){
        EStep();
        std::cout<<"e step done"<<std::endl;
        this->m_gamma2 = (this->m_gamma.array() / (this->m_gamma.rowwise().sum()).replicate(1, this->m_data_pos.cols()).array()).matrix();
        MStep();
        std::cout<<"m step done"<<std::endl;

        LL.push_back(this->m_L.colwise().sum().array().log().sum() / this->m_data_pos.cols()); //daviates by 0.005 from MATLAB code
        if(iter>=this->m_minIterEM && (LL[iter] - LL[iter - 1] < this->m_maxDiffLL || iter == this->m_maxIterEM - 1)){
            std::cout<< "Converged after "<<std::to_string(iter) <<" iterations. "<<std::endl;
            return;
        }
    }
    std::cout<<" The maximum number of iterations has been reached."<<std::endl;
}

// Checked!
void GMM::GMR(MatrixXd *xd, std::vector<MatrixXd> *sigmaXd){
    double regTerm = 1e-8;
    MatrixXd expData(2,this->m_n);
    expData.setZero();
    MatrixXd expSigma(2,2);
    MatrixXd muTmp(this->m_dimOut, this->m_k);
    MatrixXd HTmp(this->m_k, this->m_nData);

    for(int t=0; t<this->m_n;t++){
        double DataIn = (t+1)*this->m_dt;
        muTmp.setZero();
        HTmp.setZero();
        expSigma.setZero();

        // Compute activation weights
        for(int k=0;k<this->m_k;k++){
            HTmp(k,t) = this->m_priors[k] * GaussPDF(DataIn, this->m_mu(0,k), this->m_sigma[k](0,0));
        }
        HTmp.col(t) = (HTmp.col(t).array() / (HTmp.col(t).array()+std::numeric_limits<double>::min()).sum()).matrix();

        //Compute conditional means
        for(int k=0;k<this->m_k;k++){
//                std::cout<<"here0"<<std::endl;
//                std::cout<<this->m_mu.col(k).bottomRows(2)<<std::endl;
//                std::cout<<"here0"<<std::endl;
//                std::cout<<this->m_sigma[k].leftCols(1).bottomRows(2)<<std::endl;
//                std::cout<<"here0"<<std::endl;
//                std::cout<<this->m_sigma[k](0,0)<<std::endl;
//                std::cout<<"here0"<<std::endl;
//                std::cout<<(DataIn-this->m_mu(0,k))<<std::endl;
            muTmp.col(k) = this->m_mu.col(k).bottomRows(2) + (this->m_sigma[k].leftCols(1).bottomRows(2).array() / this->m_sigma[k](0,0)*(DataIn-this->m_mu(0,k))).matrix();
//                std::cout<<"here"<<std::endl;
//                std::cout<<muTmp.col(k)<<std::endl;
//                std::cout<<"here"<<std::endl;
//                std::cout<<HTmp(k,t)<<std::endl;
//                std::cout<<"here"<<std::endl;
//                std::cout<<expData.col(t)<<std::endl;
//                std::cout<<"here --->"<<std::endl;
//                std::cout<<expData.col(t) + HTmp(k,t) * muTmp.col(k)<<std::endl;
            expData.col(t) = expData.col(t) + HTmp(k,t) * muTmp.col(k);
//                std::cout<<"expData.col(t)"<<std::endl;
//                std::cout<<expData.col(t)<<std::endl;
        }
        //Compute conditional covariances
        for(int k=0;k<this->m_k;k++){
//                std::cout<<"here0"<<std::endl;
//                std::cout<<this->m_sigma[k].block(1,1,2,2)<<std::endl;
//                std::cout<<"here0"<<std::endl;
//                std::cout<<this->m_sigma[k].bottomRows(2).leftCols(1)<<std::endl;
//                std::cout<<"here0"<<std::endl;
//                std::cout<<this->m_sigma[k](0,0)<<std::endl;
//                std::cout<<"here0"<<std::endl;
//                std::cout<<this->m_sigma[k].topRows(1).rightCols(2)<<std::endl;
            MatrixXd sigmaTmp = this->m_sigma[k].block(1,1,2,2) - this->m_sigma[k].bottomRows(2).leftCols(1) / this->m_sigma[k](0,0) * this->m_sigma[k].topRows(1).rightCols(2);
//                std::cout<<"----"<<std::endl;
//                std::cout<<sigmaTmp<<std::endl;
//                std::cout<<"here2"<<std::endl;
//                std::cout<<sigmaTmp + muTmp.col(k)*muTmp.col(k).transpose()<<std::endl;
//                std::cout<<"here2"<<std::endl;
//                std::cout<<HTmp(k,t)<<std::endl;
//                std::cout<<"here2"<<std::endl;
//                std::cout<<expSigma[t]<<std::endl;
//                std::cout<<"here2"<<std::endl;

                expSigma = (expSigma + HTmp(k,t)*(sigmaTmp + muTmp.col(k)*muTmp.col(k).transpose()));
        }
        expSigma = expSigma - expData.col(t)*expData.col(t).transpose() + (MatrixXd(this->m_dimOut, this->m_dimOut).setIdentity()*regTerm);
        (*xd).col(t)=expData.col(t);
        (*sigmaXd).push_back(expSigma);
    }
}