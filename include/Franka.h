#ifndef MA_THESIS_FRANKA_H
#define MA_THESIS_FRANKA_H

#include <dqrobotics/DQ.h>
#include <dqrobotics/interfaces/vrep/DQ_VrepInterface.h>
#include <dqrobotics/utils/DQ_LinearAlgebra.h>
#include <dqrobotics/utils/DQ_Constants.h>
#include <dqrobotics/robot_modeling/DQ_SerialManipulatorDH.h>
#include <random>
//#include "Mapping_utils.h"
#include "utils.h"
#include <chrono>
#include <thread>

using namespace Eigen;
using namespace std;

class Franka {

    MatrixXd m_kinematics;
    DQ_VrepInterface m_vi;
    vector<std::string> m_jointNames;
    VectorXd m_qt;
    bool m_useVREP;

public:
    Franka(bool useVREP);
    bool usingVREP();
    DQ_VrepInterface getVREPInterface();
    void startSimulation();
    void stopSimulation();
    int m_dof;
    void moveToQGoal(const VectorXd& q_goal);
    DQ_SerialManipulator getKinematicsDQ();
    MatrixXd GetVelocityConstraints();
    MatrixXd GetJointConstraints();
    MatrixXd GetRandomJointConfig(int n);
    vector<MatrixXd> ComputeJointDerivative(const MatrixXd& J);
    MatrixXd ComputeManipulabilityJacobian(const MatrixXd& J);
    MatrixXd getManipulabilityFromVI();
    MatrixXd getTranslationJacobian();
    MatrixXd getTranslationJacobian(const MatrixXd &q);
    MatrixXd getRotationJacobian();
    MatrixXd getPoseJacobian(const MatrixXd &q);
    MatrixXd getPoseJacobian();
    MatrixXd buildGeometricJacobian(MatrixXd J, MatrixXd qt);
    void setJointPositions(VectorXd q);
    MatrixXd getManipulabilityMajorAxis(const MatrixXd& m);
    MatrixXd getManipulabilityLength(const MatrixXd& m);
    void StopSimulation();
    VectorXd getCurrentJointPositions();
    DQ getCurrentPositionDQ(const MatrixXd &q);
    VectorXd getCurrentPosition(const MatrixXd &q);
    VectorXd getCurrentPosition();
    std::vector<MatrixXd> ComputeTensorMatrixProduct(const vector<MatrixXd>& T, const MatrixXd& U, int mode);
    };


#endif //MA_THESIS_FRANKA_H
