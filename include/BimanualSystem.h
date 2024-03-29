#include "Franka.h"
#include "utils.h"

using namespace Eigen;
using namespace std;

class BimanualSystem{
public:
    double m_distance; //cm
    Franka m_leftArm = Franka(false);
    Franka m_rightArm = Franka(false);
    MatrixXd m_graspMatrix;
    MatrixXd m_blockJacobian;

    BimanualSystem();
    MatrixXd getManipulability();
    MatrixXd ComputeManipulabilityJacobian(const MatrixXd& J1Full, const MatrixXd &J2Full, const MatrixXd &G);

};