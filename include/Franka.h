#ifndef MA_THESIS_FRANKA_H
#define MA_THESIS_FRANKA_H

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/QR>
#include <math.h>
#include <dqrobotics/DQ.h>
#include <dqrobotics/interfaces/vrep/DQ_VrepInterface.h>
#include <dqrobotics/utils/DQ_LinearAlgebra.h>
#include <dqrobotics/utils/DQ_Constants.h>
#include<dqrobotics/robot_modeling/DQ_SerialManipulatorDH.h>

using Eigen::MatrixXd;
using namespace Eigen;
using namespace std;

class Franka {

    MatrixXd m_kinematics;
    DQ_VrepInterface m_vi;
    std::vector<std::string> m_jointNames;
//    DQ_SerialManipulator m_robot;
public:
    Franka();
    void moveToQGoal(VectorXd q_goal);
    void ManipulabilityTrackingMainTask(MatrixXd goal);
    void ManipulabilityTrackingSecondaryTask(MatrixXd goal);
    DQ_SerialManipulator getKinematicsDQ();
    vector<MatrixXd> ComputeJointDerivative(MatrixXd J);
    MatrixXd ComputeManipulabilityJacobian(MatrixXd J);
    MatrixXd getManipulabilityFromVI();
    MatrixXd getTranslationJacobian();
    MatrixXd getJacobian();
    MatrixXd getManipulabilityMajorAxis(MatrixXd m);
    MatrixXd getManipulabilityLength(MatrixXd m);
    void StopSimulation();
    std::vector<MatrixXd> ComputeTensorMatrixProduct(std::vector<MatrixXd>, MatrixXd U, int mode);
};


#endif //MA_THESIS_FRANKA_H
