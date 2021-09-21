#include <chrono>
#include <thread>
#include "Franka.h"

#define deb(x) cout << #x << " " << x << endl;

Franka::Franka(){
    const double pi2 = pi/2.0;
    this->m_kinematics= MatrixXd(5,7);
    this->m_kinematics <<   0,      0,          0,          0,      0,      0,          0,
                        0.333,      0,      0.316,          0,  0.384,      0,      0.107,
                            0,      0,     0.0825,    -0.0825,      0,  0.088,     0.0003,
                         -pi2,    pi2,        pi2,       -pi2,    pi2,    pi2,          0,
                           0,       0,          0,          0,      0,      0,          0;

    this->m_vi.connect(19997,100,10);

    this->m_jointNames = {"Franka_joint1",
                            "Franka_joint2",
                            "Franka_joint3",
                            "Franka_joint4",
                            "Franka_joint5",
                            "Franka_joint6",
                            "Franka_joint7"};
//    this->m_robot = this->getKinematicsDQ();
}

DQ_SerialManipulator Franka::getKinematicsDQ(){
    return DQ_SerialManipulatorDH(this->m_kinematics, "standard");
}

void Franka::moveToQGoal(const VectorXd& q_goal){

    DQ_SerialManipulator m_robot = this->getKinematicsDQ();
    std::cout << "Starting V-REP simulation..." << std::endl;
    m_vi.start_simulation();
    DQ xd = m_robot.fkm(q_goal);
    DQ x;
    VectorXd q, u;
    MatrixXd J;
    VectorXd e(8);
    e(0)=1.0;

    std::cout << "Starting control loop..." << std::endl;
    std::cout << "Joint positions q (at starting) is: \n"<< std::endl << this->m_vi.get_joint_positions(this->m_jointNames) << std::endl;

    // Control Loop
    while(e.norm()>0.05)
    {
        // Read the current robot joint positions
        q          = m_vi.get_joint_positions(m_jointNames);
        // Perform forward kinematics to obtain current EE configuration
        x       = m_robot.fkm(q);
        // Compute error in the Task-Space
        e          = vec8(x-xd);
        // Obtain the current analytical Jacobian (Dim: 8 * n)
        J = m_robot.pose_jacobian(q);
        // Kinematic Control Law
        u = -0.01 * pinv(J) * e;
        // Integrate to obtain q_out
        q          = q + u;
        std::cout << "Tracking error: " <<e.norm()<< std::endl;
        std::cout << "====================================="<< std::endl;
        // Send commands to the robot
        m_vi.set_joint_positions(m_jointNames,q);
        // Always sleep for a while before next step
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    std::cout << "Control finished..." << std::endl;
}

void Franka::StopSimulation(){
    std::cout << "Stopping V-REP simulation..." << std::endl;
    this->m_vi.stop_simulation();
    this->m_vi.disconnect();
}

MatrixXd Franka::getTranslationJacobian() {
    DQ_SerialManipulator m_robot = this->getKinematicsDQ();
    MatrixXd q  = m_vi.get_joint_positions(m_jointNames);
    DQ x       = m_robot.fkm(q);
    MatrixXd J = m_robot.pose_jacobian(q);
    MatrixXd Jt = m_robot.translation_jacobian(J, x);
    return Jt;
}

MatrixXd Franka::getJacobian() {
    DQ_SerialManipulator m_robot = this->getKinematicsDQ();
    MatrixXd q  = m_vi.get_joint_positions(m_jointNames);
    MatrixXd J = m_robot.pose_jacobian(q);
    return J;
}

MatrixXd Franka::getManipulabilityFromVI() {
    MatrixXd JCurr = this->getTranslationJacobian();
    return JCurr*JCurr.transpose();
}

MatrixXd Franka::getManipulabilityLength(const MatrixXd& m) {
    return Eigen::MatrixXd();
}

MatrixXd Franka::getManipulabilityMajorAxis(const MatrixXd& m) {
    EigenSolver<MatrixXd> es(m);
    MatrixXd D = es.eigenvalues().real();
    MatrixXd V = es.eigenvectors().real();

    int maxIdx;
//    D.array().maxCoeff(&maxIdx);
    return V.col(maxIdx);
}

// Checked for 2d!
void Franka::ManipulabilityTrackingMainTask(const MatrixXd& MDesired) {
    int km = 3;
    int niter=65;
    float dt=1e-2;
    DQ x;
    MatrixXd J, M, qt, JmT, MDiff, pinv, dqt1;
    DQ_SerialManipulator m_robot = this->getKinematicsDQ();
//    x = m_robot.fkm(qt);

    for(int i=0;i<niter; i++){
        qt  = m_vi.get_joint_positions(m_jointNames);
        J = this->getJacobian();
        M = J*J.transpose();

        JmT = ComputeManipulabilityJacobian(J); // Checked!
        MDiff = LogMap(MDesired, M); // Checked!
        pinv = JmT.completeOrthogonalDecomposition().pseudoInverse(); // Checked!
        dqt1 = pinv*km*Symmat2Vec(MDiff).transpose(); //Checked!

        m_vi.set_joint_positions(m_jointNames,qt + dqt1*dt); // Checked!
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        std::cout << "Tracking error: " <<(MDesired.pow(-0.5)*M*MDesired.pow(-0.5)).log().norm()<< std::endl; //norm returns Frobenius norm if Matrices
        std::cout << "====================================="<< std::endl;
    }
    WriteCSV(M, "/home/nnrthmr/CLionProjects/ma_thesis/data/M.csv");
}

void Franka::ManipulabilityTrackingSecondaryTask(const MatrixXd& XDesired, const MatrixXd& DXDesired, const MatrixXd& MDesired) {
    float dt=1e-2;
    int niter=50;
    int kp=8; //gain for position control
    int km=5; // gain for manip in null space
    int nDoF = 4; //robot dofs

    DQ x;
    MatrixXd J, M, qt, JmT, DxR, pinv, dqt1, MDiff, pinv2, mnsCommand, dqns;
    MatrixXd xPos(2,1);
    DQ_SerialManipulator m_robot = this->getKinematicsDQ();

    for(int i=0;i<niter; i++) {
        qt = m_vi.get_joint_positions(m_jointNames);
        x = m_robot.fkm(qt);
        cout<<"x DQ: "<<x.vec8();
        xPos << x.translation().vec3()[0], x.translation().vec3()[1];

        J = this->getJacobian();
        M = J * J.transpose();

        JmT = ComputeManipulabilityJacobian(J); // Checked!

        // Compute joint velocities
        DxR = DXDesired + kp*(XDesired-xPos);
        pinv = J.completeOrthogonalDecomposition().pseudoInverse();
        dqt1 = pinv*DxR;

        // Compute null space joint velocities
        MDiff = LogMap(MDesired, M); // Checked!
        pinv2 = JmT.completeOrthogonalDecomposition().pseudoInverse();
        mnsCommand = pinv2 * Symmat2Vec(MDiff);
        dqns = (MatrixXd::Identity(nDoF, nDoF) - pinv*J)*km*mnsCommand; // Redundancy resolution

        m_vi.set_joint_positions(m_jointNames,qt + (dqt1+dqns)*dt);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        std::cout << "Position tracking error: " <<(XDesired-xPos).norm()<< std::endl;
        std::cout << "Manipulability tracking error: " <<(MDesired.pow(-0.5)*M*MDesired.pow(-0.5)).log().norm()<< std::endl; //norm returns Frobenius norm if Matrices
        std::cout << "====================================="<< std::endl;
    }

    string s= "/home/nnrthmr/CLionProjects/ma_thesis/data/M_Secondary.csv";
    WriteCSV(M, s);
}

std::vector<MatrixXd> Franka::ComputeTensorMatrixProduct(const vector<MatrixXd>& T, const MatrixXd& U, int mode){
//    MatrixXd sizeTens = MatrixXd::Ones(1, mode);
    MatrixXd sizeTens(1,3);
    sizeTens<< T[0].rows(), T[0].cols(), T.size();
    int N = 3;

    //Compute complement of set of modes
    MatrixXd bits(1,N);
    bits.setOnes();
    bits(0,mode-1) = 0;
    vector<int> tmp;
    for(int i=0;i<N;i++){
        if(bits(0,i)==1){
            tmp.push_back(i);
        }
    }

    ArrayXd modeC(tmp.size());
    for(int i=0;i<tmp.size();i++){
        modeC(i) = tmp[i];
    }

    // Permutation
    std::vector<int> perm;
    std::vector<int> iperm;

    perm.push_back(mode-1);
    iperm.push_back(0);
    MatrixXd sizeTensPerm(1,3);
    sizeTensPerm(0,0) = sizeTens(mode-1);
    for(int i=0;i<modeC.size();i++){
        perm.push_back(modeC(i));
        iperm.push_back(0);
        sizeTensPerm(0,i+1) = sizeTens(modeC(i));
    }

    std::vector<MatrixXd> S;
    for(int i=0;i<T.size();i++){
        S.push_back(T[i]);
    }

    if(mode!=1){
        assert(perm[0]==1 &&perm[1]==0 && perm[2]==2); // TODO not working for other perms
        for(int i=0;i<S.size();i++){
            MatrixXd tmp3=S[i].transpose();
            S[i] = tmp3;
        }
    }

    // nmode product
    sizeTensPerm(0,0) = U.rows();
    MatrixXd SReshape(S[0].rows(),S.size()*S[0].cols());
    for(int i=0;i<S.size();i++){
        SReshape.block(0,S[0].cols()*i,S[0].rows(), S[0].cols()) = S[i];
    }

    SReshape = U*SReshape;
    vector<MatrixXd> S2;
    for(int i=0;i<sizeTensPerm(0,2);i++){
        S2.push_back(SReshape.block(0,i*sizeTensPerm(0,1),sizeTensPerm(0,0), sizeTensPerm(0,1)));
    }

    for(int i=0;i<perm.size();i++){
        iperm[perm[i]] = i;
    }

    if(mode!=1){
        assert(iperm[0]==1 && iperm[1]==0 && iperm[2]==2);
        for(int i=0;i<S2.size();i++){
            MatrixXd tmp4 = S2[i].transpose();
            S2[i]=tmp4;
        }
    }
    return S2;
}

// Checked!
vector<MatrixXd> Franka::ComputeJointDerivative(const MatrixXd& J){
    std::vector<MatrixXd> JGradVec;
    MatrixXd tmp(J.rows(), J.cols());
    for(int i=0;i<J.cols();i++){
        tmp.setZero();
        JGradVec.push_back(tmp);
    }

    MatrixXd JGrad(J.rows(),J.cols());
    JGrad.setZero();
    Vector3d tmp1;
    Vector3d tmp2;

    //Compute joint derivative jacobian
    for(int i=0;i<J.cols();i++){
        for(int j=0;j<J.cols();j++) {
            if(j<i){
                tmp1= J.col(j).bottomRows(3);
                tmp2=J.col(i).topRows(3);
                JGradVec[j].block(0,i,3,1) = tmp1.cross(tmp2);
                tmp1=J.col(j).bottomRows(3);
                tmp2=J.col(i).bottomRows(3);
                JGradVec[j].block(3,i,3,1) = tmp1.cross(tmp2);
            }
            if(j>i){
                tmp1 =J.col(j).topRows(3);
                tmp2=J.col(i).bottomRows(3);
                JGradVec[j].block(0,i,3,1) = -(tmp1.cross(tmp2));
            }
            else{
                tmp1=J.col(i).bottomRows(3);
                tmp2 = J.col(i).topRows(3);
                JGradVec[j].block(0,i,3,1) = tmp1.cross(tmp2);
            }
        }
    }
    return JGradVec;
}

// Checked!
MatrixXd Franka::ComputeManipulabilityJacobian(const MatrixXd& J){
    MatrixXd W = MatrixXd::Identity(J.cols(), J.cols());
    std::vector<MatrixXd> JGradVec;
    JGradVec = ComputeJointDerivative(J);
    std::vector<MatrixXd> Jm;
    MatrixXd tmp, tmp3;
    MatrixXd U= J*(W*W.transpose());
    vector<MatrixXd> tmp1= ComputeTensorMatrixProduct(JGradVec, U, 2);

    for(int i=0;i<JGradVec.size();i++) {
        tmp = JGradVec[i].transpose();
        JGradVec[i] = tmp;
    }

    vector<MatrixXd> tmp2 = ComputeTensorMatrixProduct(JGradVec, U, 1);
    for(int i=0;i<tmp1.size();i++) {
        tmp3 = tmp1[i] + tmp2[i];
        Jm.push_back(tmp3);
    }

    MatrixXd JmRed(3, Jm.size());
    JmRed.setZero();
    for(int i=0;i<Jm.size();i++)
        JmRed.col(i)=(Symmat2Vec(Jm[i].block(0,0,2,2))).transpose(); //TaskVar 1:2 in MATLAB
    return JmRed;
}
