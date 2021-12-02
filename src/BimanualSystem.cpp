#include "BimanualSystem.h"

BimanualSystem::BimanualSystem(){
    this->m_distance = 0.0; //cm

    this->m_graspMatrix=MatrixXd(3,6);
    this->m_graspMatrix <<  1,0,0,1,0,0,
                            0,1,0,0,1,0,
                            0,0,1,0,0,1;

    this->m_blockJacobian=MatrixXd(12,14);
    this->m_blockJacobian.setZero();
    VectorXd qt1  = this->m_leftArm.getCurrentJointPositions();
    MatrixXd Jpose1= this->m_leftArm.getPoseJacobian();
    MatrixXd Jgeo1 = this->m_leftArm.buildGeometricJacobian(Jpose1, qt1);
    VectorXd qt2  = this->m_rightArm.getCurrentJointPositions();
    MatrixXd Jpose2= this->m_rightArm.getPoseJacobian();
    MatrixXd Jgeo2 = this->m_rightArm.buildGeometricJacobian(Jpose1, qt2);
    this->m_blockJacobian.block(0,0,6,7) = Jgeo1;
    this->m_blockJacobian.block(6,7,6,7) = Jgeo2;
}

MatrixXd BimanualSystem::getManipulability(){
    MatrixXd pinvG = pinv(this->m_graspMatrix);
    return (pinvG.transpose()*this->m_blockJacobian*this->m_blockJacobian.transpose()*pinvG).inverse();
}

// Checked!
MatrixXd BimanualSystem::ComputeManipulabilityJacobian(const MatrixXd& J1Full, const MatrixXd &J2Full, const MatrixXd &G){ //JFull = getPoseJacobian()
    int ndof1=J1Full.cols();
    int ndof2=J2Full.cols();

    MatrixXd J(6, ndof1+ndof2);
    J.setZero();
    J.block(0,0,3,ndof1) = J1Full.topRows(3);
    J.block(3,ndof1,3,ndof2) = J2Full.topRows(3);

    vector<MatrixXd> JGradVec1;
    vector<MatrixXd> JGradVec2;
    JGradVec1 = this->m_leftArm.ComputeJointDerivative(J1Full);
    JGradVec2 = this->m_rightArm.ComputeJointDerivative(J2Full);

    vector<MatrixXd> JGrad;

    for(int i=0;i<ndof1+ndof2;++i){
        JGrad.push_back(MatrixXd(6, ndof1+ndof2).setZero());
        if(i<ndof1) JGrad[i].block(0,0,3,ndof1) = JGradVec1[i].topRows(3);
        else JGrad[i].block(3,ndof1,3,ndof2) = JGradVec2[i-ndof1].topRows(3);
    }

    MatrixXd pG = pinv(G);

    vector<MatrixXd> Jm;
    vector<MatrixXd> tmp1= this->m_leftArm.ComputeTensorMatrixProduct(JGrad, pG.transpose(), 1);
    vector<MatrixXd> tmp2= this->m_leftArm.ComputeTensorMatrixProduct(tmp1, pG.transpose()*J, 2);

    //eqvalent of permute operation in matlab code
    MatrixXd tmp;
    for(int i=0;i<JGrad.size();i++) {
        tmp = JGrad[i].transpose();
        JGrad[i] = tmp;
    }

    vector<MatrixXd> tmp3 = this->m_leftArm.ComputeTensorMatrixProduct(JGrad, pG.transpose()*J, 1);
    vector<MatrixXd> tmp4 = this->m_leftArm.ComputeTensorMatrixProduct(tmp3, pG.transpose(), 2);

    MatrixXd tmp5;
    for(int i=0;i<tmp1.size();i++) {
        tmp5 = tmp2[i] + tmp4[i];
        Jm.push_back(tmp5);
    }

    MatrixXd JmRed=MatrixXd(6, Jm.size());
    JmRed.setZero();
    for(int i=0;i<Jm.size();i++) {
        JmRed.col(i) = (symmat2Vec(Jm[i])).transpose(); //TaskVar 1:3 in MATLAB
    }

    return JmRed;

}


