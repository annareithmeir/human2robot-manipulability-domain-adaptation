#include "GMM_SPD.h"
#include <Eigen/Core>

GMM_SPD::GMM_SPD() {
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

void GMM_SPD::CumulativeSum(const Eigen::VectorXd& input, Eigen::VectorXd& result) {
    result(0)=input[0];
    for (int i = 1; i < input.size(); i++) {
        result(i) = result(i - 1) + input(i);
    }
}

//Checked!
void GMM_SPD::InitModel(Eigen::MatrixXd *data){
    this->m_n = data->rows() / this->m_nDemos; //nbData
    this->m_mu= MatrixXd(this->m_dimVar, this->m_k);

    std::vector<int> timing = linspace(0, this->m_n, this->m_k+1);
    for(int i=0;i<timing.size();i++){
        std::cout<<timing[i]<<std::endl;
    }



    for(int i=0; i<this->m_k;i++){
        std::vector<int> collected;
        for(int d=0;d<this->m_nDemos; d++) {
            for(int t=timing[i];t<timing[i+1]; t++) {
            collected.push_back(d * this->m_n + t);
            }
        }
        this->m_priors.push_back(collected.size());
        MatrixXd collectedMatrix(collected.size(),3);
        for(int l=0; l<collected.size();l++){
            collectedMatrix.block(l,0,1,3) = (*data).row(collected[l]).rightCols(3);
        }

//        std::cout<<collectedMatrix<<std::endl;
//        std::cout<<std::endl;

//        this->m_mu.col(i) = Symmat2Vec(SPDMean(Vec2Symmat(collectedMatrix)));

//        MatrixXd centered = collected.colwise() - collected.rowwise().mean();
//        MatrixXd cov = (centered * centered.adjoint()) / double(collected.cols() - 1);
//        this->m_sigma.push_back(cov);
    }
    double priorsSum = (double) (std::accumulate(this->m_priors.begin(), this->m_priors.end(), 0.0f));
    for(double& d : this->m_priors){
        d /= priorsSum;
    }

    this->m_L = MatrixXd(this->m_k, data->cols());
    this->m_gamma = MatrixXd(this->m_k, data->cols());
    this->m_gamma2 = MatrixXd(this->m_k, data->cols());
}

Eigen::MatrixXd GMM_SPD::SPDMean() {
    return Eigen::MatrixXd();
}

Eigen::VectorXd GMM_SPD::Symmat2Vec(Eigen::MatrixXd mat) {
    return Eigen::VectorXd();
}

std::vector<Eigen::MatrixXd> GMM_SPD::Vec2Symmat(Eigen::MatrixXd vec) {
    std::vector<Eigen::MatrixXd> MVector;
//    if(vec.cols()==1 || vec.rows()==1){
//        std::cout<<"1111"<<std::endl;
//        int n=vec.cols();
//        if(n==1) {
//            n=vec.rows();
//            vec=vec.transpose(); //1xn
//        }
//        std::cout<<"1"<<std::endl;
//        int N = (-1+ sqrt(1+8*n))/2;
//        Eigen::MatrixXd M = vec.row(0).leftCols(N).asDiagonal();
//        std::cout<<"1"<<std::endl;
//        Eigen::VectorXd id(N);
//        CumulativeSum(Eigen::VectorXd::LinSpaced(N, N-1, 0), id);
//        std::cout<<"----"<<std::endl;
//        std::cout<<n<<std::endl;
//        std::cout<<N<<std::endl;
//        std::cout<<M<<std::endl;
//        std::cout<<id<<std::endl;
//        for(int i=0;i<N-1;i++){
////            MatrixXd tmp= vec.block(id(i)+1,i,id(i+1)-id(i)+1, 1).asDiagonal() * (1/sqrt(2)) + vec.block(id(i)+1,-i,id(i+1)-id(i)+1, 1).asDiagonal() * (1/sqrt(2));
////            M=M+tmp;
//        }
//        MVector.push_back(M);
//    }
//    else{
        std::cout<<"222"<<std::endl;
        int d= vec.rows();
        int N = vec.cols();
        int D = (-1+ sqrt(1+8*d))/2;
        int row;
        std::cout<<D<<" "<<d<<" "<<N<<std::endl;
        for(int i=0;i<N;i++){ //colwise
            std::cout<<i<<std::endl;
            Eigen::MatrixXd vn = vec.col(i).transpose();
            Eigen::MatrixXd Mn= vn.row(0).leftCols(D).asDiagonal();
            std::cout<<"----mn -----"<<Mn<<std::endl;
            Eigen::VectorXd id(D);
            CumulativeSum(Eigen::VectorXd::LinSpaced( D,  D, 1), id);
            std::cout<<"----id -----"<<id<<std::endl;

            Eigen::MatrixXd tmp1(Mn.rows(), Mn.cols());
            Eigen::MatrixXd tmp2(Mn.rows(), Mn.cols());
            for(int j=0;j<D-1;j++){
                std::cout<<"----j -----"<<j<<std::endl;
                tmp1.setZero();
                row=0;             // TODO in second iteration row is set to 0 again, must start at 2 to fill matrix...
                for(int k=i;k<id(i+1)-id(i);k++){
                    tmp1(row,k+1) = vn(0,id(i)+row)* (1/sqrt(2));
//                    tmp1(row,k+1) = vn(0,id(i)+1+row)* (1/sqrt(2));
                    row++;
                }
                std::cout<<"----tmp1 -----"<<tmp1<<std::endl;
                tmp2.setZero();
                row=0;
                for(int k=i;k<id(i+1)-id(i);k++){
                    tmp2(k+1,row) = vn(0,id(i)+row)* (1/sqrt(2));
//                    tmp2(row+k+1,k) = vn(0,id(i)+1+row)* (1/sqrt(2));
                    row++;
                }
                std::cout<<"----tmp2 -----"<<tmp2<<std::endl;
                Mn=Mn+tmp1+tmp2;
                std::cout<<"----Mn -----"<<std::endl;
                std::cout<<Mn<<std::endl;
            }
            MVector.push_back(Mn);
        }
//    }
    return MVector;
}

//Checked! (numerical slightly different
Eigen::VectorXd GMM_SPD::GaussPDF(Eigen::MatrixXd mu, Eigen::MatrixXd sig){
    Eigen::MatrixXd pdf(1, this->m_n);
    Eigen::MatrixXd dataCentered = this->m_data_pos.transpose() - mu.transpose().replicate(this->m_data_pos.cols(),1);
    Eigen::MatrixXd tmp = dataCentered*(sig.inverse());
    pdf = (tmp.array() * dataCentered.array()).matrix().rowwise().sum();
    pdf = (-0.5*pdf).array().exp() / sqrt(pow(2*M_PI, this->m_dimVar)*abs(sig.determinant()));
    return pdf;
}

void GMM_SPD::EStep() {
    for(int k=0; k< this->m_k; k++){
        this->m_L.row(k) = this->m_priors[k] * GaussPDF(this->m_mu.col(k), this->m_sigma[k]).transpose();
    }
    this->m_gamma = (this->m_L.array() / (this->m_L.colwise().sum().array()+std::numeric_limits<double>::min()).replicate(this->m_k, 1).array()).matrix();
}

//Checked!
void GMM_SPD::MStep() {
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

void GMM_SPD::TrainEM(){
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

