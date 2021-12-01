#import "BimanualSystem.h"

BimanualSystem::BimanualSystem(){
    this->m_distance = 0.0; //cm

    this->m_graspMatrix(3,6);
    this->m_graspMatrix <<  1,0,0,1,0,0,
                            0,1,0,0,1,0,
                            0,0,1,0,0,1;

    this->m_blockJacobian(12,14);
    this->m_blockJacobian.setZero();
    this->m_blockJacobian.block(0,0,6,7) = this->m_leftArm.getPoseJacobian();
    this->m_blockJacobian.block(6,7,6,7) = this->m_rightArm.getPoseJacobian();
}

MatrixXd BimanualSystem::getManipulability(){
    MatrixXd pinvG = pinv(this->m_graspMatrix);
    return (pinvG.transpose()*this->m_blockJacobian*this->m_blockJacobian.transpose()*pinvG).inverse();
}

MatrixXd BimanualSystem::ComputeManipulabilityJacobian(const MatrixXd& J1Full, const MatrixXd &J2Full){ //JFull = getPoseJacobian()
    return MatrixXd(1,1);
}


