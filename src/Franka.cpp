#include <chrono>
#include <thread>
#include "Franka.h"


Franka::Franka(){
    this->m_kinematics= MatrixXd(5,7);

    const double pi2 = pi/2.0;
    this->m_kinematics <<   0,      0,          0,          0,      0,      0,          0,
                        0.333,      0,      0.316,          0,  0.384,      0,      0.107,
                            0,      0,     0.0825,    -0.0825,      0,  0.088,     0.0003,
                         -pi2,    pi2,        pi2,       -pi2,    pi2,    pi2,          0,
                           0,       0,          0,          0,      0,      0,          0;


//    this->m_kinematicsDQ = dqKin;
}

DQ_SerialManipulator Franka::getKinematicsDQ(){
    return DQ_SerialManipulatorDH(this->m_kinematics, "standard");
}

void Franka::moveToQGoal(VectorXd q_goal){
    // Object from DQRobotics class for communication with VREp
    DQ_VrepInterface vi;
    // Open the port and connect to VREP
    vi.connect(19997,100,10);
    std::cout << "Starting V-REP simulation..." << std::endl;
    vi.start_simulation();
    // Name robot joint handles for the vrep function
    std::vector<std::string> joint_names = {"Franka_joint1",
                                            "Franka_joint2",
                                            "Franka_joint3",
                                            "Franka_joint4",
                                            "Franka_joint5",
                                            "Franka_joint6",
                                            "Franka_joint7"};
    // Construct the kinematic model of the robot
//    Franka franka=Franka();
    DQ_SerialManipulator robot = this->getKinematicsDQ();
    // Initialize the goal position in the joint space
//    VectorXd q_goal(7);
//    q_goal << -pi/2.0, 0.004, 0.0, -1.57156, 0.0, 1.57075, 0.0;
    // Goal in the task space represented by a DQ
    DQ xd = robot.fkm(q_goal);

    VectorXd q;
    VectorXd e(8);
    e(0)=1.0;

    std::cout << "Starting control loop..." << std::endl;
    std::cout << "Joint positions q (at starting) is: \n"<< std::endl << vi.get_joint_positions(joint_names) << std::endl;

    // Control Loop
    while(e.norm()>0.05)
    {
        // Read the current robot joint positions
        q          = vi.get_joint_positions(joint_names);
        // Perform forward kinematics to obtain current EE configuration
        DQ x       = robot.fkm(q);
        // Compute error in the Task-Space
        e          = vec8(x-xd);
        // Obtain the current analytical Jacobian (Dim: 8 * n)
        MatrixXd J = robot.pose_jacobian(q);
        // Kinematic Control Law
        VectorXd u = -0.01 * pinv(J) * e;
        // Integrate to obtain q_out
        q          = q + u;
        std::cout << "Tracking error: " <<e.norm()<< std::endl;
        std::cout << "====================================="<< std::endl;
        // Send commands to the robot
        vi.set_joint_positions(joint_names,q);
        // Always sleep for a while before next step
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    std::cout << "Control finished..." << std::endl;
    std::cout << "Stopping V-REP simulation..." << std::endl;
    vi.stop_simulation();
    vi.disconnect();
}

MatrixXd Franka::getManipulabilityFromVI() {

    return Eigen::MatrixXd();
}

MatrixXd Franka::getManipulabilityLength(MatrixXd m) {
    return Eigen::MatrixXd();
}

MatrixXd Franka::getManipulabilityMajorAxis(MatrixXd m) {
    EigenSolver<MatrixXd> es(m);
    MatrixXd D = es.eigenvalues().real();
    MatrixXd V = es.eigenvectors().real();

    int maxIdx;
//    D.array().maxCoeff(&maxIdx);
    return V.col(maxIdx);
}
