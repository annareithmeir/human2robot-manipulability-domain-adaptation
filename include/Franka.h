#ifndef MA_THESIS_FRANKA_H
#define MA_THESIS_FRANKA_H

#include <Eigen/Core>
#include <math.h>
#include <dqrobotics/DQ.h>
#include <dqrobotics/interfaces/vrep/DQ_VrepInterface.h>
#include <dqrobotics/utils/DQ_LinearAlgebra.h>
#include <dqrobotics/utils/DQ_Constants.h>
#include<dqrobotics/robot_modeling/DQ_SerialManipulatorDH.h>

using Eigen::MatrixXd;
using namespace Eigen;

class Franka {

    MatrixXd m_kinematics;

public:
    Franka();
    void moveToQGoal(VectorXd q_goal);
    DQ_SerialManipulator getKinematicsDQ();
    MatrixXd getManipulabilityFromVI();
    MatrixXd getManipulabilityMajorAxis(MatrixXd m);
    MatrixXd getManipulabilityLength(MatrixXd m);
};


#endif //MA_THESIS_FRANKA_H
