#ifndef MA_THESIS_FRANKA_H
#define MA_THESIS_FRANKA_H

#include <dqrobotics/DQ.h>
#include <dqrobotics/interfaces/vrep/DQ_VrepInterface.h>
#include <dqrobotics/utils/DQ_LinearAlgebra.h>
#include <dqrobotics/utils/DQ_Constants.h>
#include <dqrobotics/robot_modeling/DQ_SerialManipulatorDH.h>
#include "SPD_Utils.h"

using namespace Eigen;
using namespace std;

class Franka {

    MatrixXd m_kinematics;
    DQ_VrepInterface m_vi;
    std::vector<std::string> m_jointNames;
//    DQ_SerialManipulator m_robot;
public:
    Franka();
    void startSimulation();
    void stopSimulation();
    int m_dof;
    void moveToQGoal(const VectorXd& q_goal);
    MatrixXd ManipulabilityTrackingMainTask(const MatrixXd& goal, vector<MatrixXd> &mLoop, vector<double> &eLoop);
    MatrixXd ManipulabilityTrackingSecondaryTask(const MatrixXd& XDesired, const MatrixXd& DXDesired, const MatrixXd& MDesired);
    DQ_SerialManipulator getKinematicsDQ();
    MatrixXd GetVelocityConstraints();
    MatrixXd GetJointConstraints();
    vector<MatrixXd> ComputeJointDerivative(const MatrixXd& J);
    MatrixXd ComputeManipulabilityJacobian(const MatrixXd& J);
    MatrixXd getManipulabilityFromVI();
    MatrixXd getTranslationJacobian();
    MatrixXd getRotationJacobian();
    MatrixXd getTranslationJacobian(MatrixXd q);
    MatrixXd getPoseJacobian(MatrixXd q);
    MatrixXd getJacobian();
    MatrixXd buildGeometricJacobian(MatrixXd J, MatrixXd qt);
    void setJoints(VectorXd q);
    MatrixXd getManipulabilityMajorAxis(const MatrixXd& m);
    MatrixXd getManipulabilityLength(const MatrixXd& m);
    void StopSimulation();
    VectorXd getCurrentJointPositions();
    DQ getCurrentPositionDQ(MatrixXd q);
    VectorXd getCurrentPosition(MatrixXd q);
    VectorXd getCurrentPosition();
    std::vector<MatrixXd> ComputeTensorMatrixProduct(const vector<MatrixXd>& T, const MatrixXd& U, int mode);

};


#endif //MA_THESIS_FRANKA_H
