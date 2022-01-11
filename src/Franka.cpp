#include "Franka.h"

#define dimensions 3
#define USE_VREP false

Franka::Franka(bool useVREP){
    if(useVREP) this->m_useVREP=true;
    else this->m_useVREP=false;

    const double pi2 = pi/2.0;
    this->m_kinematics= MatrixXd(5,7);
    this->m_kinematics <<   0,      0,          0,          0,      0,      0,          0,
                        0.333,      0,      0.316,          0,  0.384,      0,      0.107,
                            0,      0,     0.0825,    -0.0825,      0,  0.088,     0.0003,
                         -pi2,    pi2,        pi2,       -pi2,    pi2,    pi2,          0,
                           0,       0,          0,          0,      0,      0,          0;

    if(USE_VREP) this->m_vi.connect(19997,100,10);
    this->m_dof=7;

    this->m_jointNames = {"Franka_joint1",
                            "Franka_joint2",
                            "Franka_joint3",
                            "Franka_joint4",
                            "Franka_joint5",
                            "Franka_joint6",
                            "Franka_joint7"};

    this->m_qt = VectorXd(7);
    this->m_qt << +6.600e+01, +2.200e+01, +1.500e+01, -9.000e+01, +0.000e+00, +8.000e+01, +0.000e+00;
}

void Franka::startSimulation(){
    m_vi.start_simulation();
}

void Franka::stopSimulation(){
    std::cout << "Stopping V-REP simulation..." << std::endl;
    this->m_vi.stop_simulation();
    this->m_vi.disconnect();
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
    vector<VectorXd> qs;

    std::cout << "Starting control loop..." << std::endl;
    std::cout << "Joint positions q (at starting) is: \n"<< std::endl << getCurrentJointPositions() << std::endl;

    // Control Loop
    while(e.norm()>0.05)
    {
        // Read the current robot joint positions
        q          = getCurrentJointPositions();
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
        setJointPositions(q);
        qs.push_back(q);
        // Always sleep for a while before next step
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    std::cout << "Control finished..." << std::endl;
    MatrixXd qs_m(qs.size(), 7);
    for(int i=0;i<qs.size();i++){
        qs_m.row(i) = qs[i];
    }

    writeCSV(qs_m, "/home/nnrthmr/CLionProjects/ma_thesis/data/demos/towards_singularities/panda/panda_reach_up_joints.csv");
}

void Franka::setJointPositions(VectorXd q){
    if(m_useVREP) m_vi.set_joint_positions(m_jointNames,q);
    m_qt = q;
}

VectorXd Franka::getCurrentJointPositions(){
    return m_qt;
}

MatrixXd Franka::getTranslationJacobian() {
    DQ_SerialManipulator m_robot = this->getKinematicsDQ();
    MatrixXd q  = getCurrentJointPositions();
    DQ x       = m_robot.fkm(q);
    MatrixXd J = m_robot.pose_jacobian(q);
    MatrixXd Jt = m_robot.translation_jacobian(J, x);
    return Jt;
}

MatrixXd Franka::getRotationJacobian() {
    DQ_SerialManipulator m_robot = this->getKinematicsDQ();
    MatrixXd q  = getCurrentJointPositions();
    DQ x       = m_robot.fkm(q);
    MatrixXd J = m_robot.pose_jacobian(q);
    MatrixXd Jt = m_robot.rotation_jacobian(J);
    return Jt;
}

MatrixXd Franka::getTranslationJacobian(const MatrixXd &q) {
    DQ_SerialManipulator m_robot = this->getKinematicsDQ();
    DQ x       = m_robot.fkm(q);
    MatrixXd J = m_robot.pose_jacobian(q);
    MatrixXd Jt = m_robot.translation_jacobian(J, x);
    return Jt;
}

MatrixXd Franka::getPoseJacobian(const MatrixXd &q) {
    DQ_SerialManipulator m_robot = this->getKinematicsDQ();
    DQ x       = m_robot.fkm(q);
    MatrixXd J = m_robot.pose_jacobian(q);
    return J;
}

DQ Franka::getCurrentPositionDQ(const MatrixXd &q) {
    DQ_SerialManipulator m_robot = this->getKinematicsDQ();
    return m_robot.fkm(q);
}

VectorXd Franka::getCurrentPosition(const MatrixXd &q) {
    DQ_SerialManipulator m_robot = this->getKinematicsDQ();
    DQ x = m_robot.fkm(q);
    VectorXd xvec = vec3(x);
    return xvec;
}

VectorXd Franka::getCurrentPosition() {
    DQ_SerialManipulator m_robot = this->getKinematicsDQ();
    MatrixXd q          = getCurrentJointPositions();
    // Perform forward kinematics to obtain current EE configuration
    DQ x       = m_robot.fkm(q);
    Vector3d xvec;
    xvec << x.translation().vec3()[0],x.translation().vec3()[1],x.translation().vec3()[2];
    return xvec;
}

MatrixXd Franka::getPoseJacobian() {
    DQ_SerialManipulator m_robot = this->getKinematicsDQ();
    MatrixXd q  = getCurrentJointPositions();
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
vector<MatrixXd> Franka::ComputeJointDerivative(const MatrixXd& J){ // How to calculate this for Jred or DQ Jfull?
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

   if(dimensions==2){
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
   }
   else{
       //Compute joint derivative jacobian --> Checked!
       for(int i=0;i<J.cols();i++){
           for(int j=0;j<J.cols();j++) {
               if(j<i){
                   tmp1= J.block(3,j,3,1);
                   tmp2=J.block(0,i,3,1);
                   JGradVec[j].block(0,i,3,1) = tmp1.cross(tmp2);
                   tmp1=J.block(3,j,3,1);
                   tmp2=J.block(3,i,3,1);
                   JGradVec[j].block(3,i,3,1) = tmp1.cross(tmp2);
               }
               else if(j>i){
                   tmp1 =J.block(0,j,3,1);
                   tmp2=J.block(3,i,3,1);
                   JGradVec[j].block(0,i,3,1) = -(tmp1.cross(tmp2));
               }
               else{
                   tmp1=J.block(3,i,3,1);
                   tmp2 = J.block(0,i,3,1);
                   JGradVec[j].block(0,i,3,1) = tmp1.cross(tmp2);
               }
           }
       }
   }
    return JGradVec;
}

// Checked!
MatrixXd Franka::ComputeManipulabilityJacobian(const MatrixXd& J){
    MatrixXd W = MatrixXd::Identity(J.cols(), J.cols());
    vector<MatrixXd> JGradVec;
    JGradVec = ComputeJointDerivative(J);

    vector<MatrixXd> Jm;
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

    MatrixXd JmRed;
    if(dimensions==2) JmRed=MatrixXd(3, Jm.size());
    else JmRed=MatrixXd(6, Jm.size());
    JmRed.setZero();
    for(int i=0;i<Jm.size();i++) {
        if (dimensions==2) JmRed.col(i) = (symmat2Vec(Jm[i].block(0, 0, 2, 2))).transpose(); //TaskVar 1:2 in MATLAB
        else JmRed.col(i) = (symmat2Vec(Jm[i].block(0, 0, 3, 3))).transpose(); //TaskVar 1:3 in MATLAB
    }
    return JmRed;
}

MatrixXd Franka::ComputeManipulabilityJacobianLower(const MatrixXd& J){
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

    MatrixXd JmRed;
    if(dimensions==2) JmRed=MatrixXd(3, Jm.size());
    else JmRed=MatrixXd(6, Jm.size());
    JmRed.setZero();
    for(int i=0;i<Jm.size();i++) {
        JmRed.col(i) = (symmat2Vec(Jm[i].block(3, 3, 3, 3))).transpose(); //TaskVar 4:6 in MATLAB
    }
    return JmRed;
}

// Checked
MatrixXd Franka::buildGeometricJacobian(MatrixXd J, MatrixXd qt){
    MatrixXd J6(6, J.cols());
    J6.setZero();

    DQ_SerialManipulator m_robot = this->getKinematicsDQ();
    DQ x = m_robot.fkm(qt);

    MatrixXd tmp(3,4);
    MatrixXd tmp2(6, J.cols());
    MatrixXd tmp3(4,4);
    MatrixXd J8(8, J.cols());
    tmp.setZero();
    tmp2.setZero();
    tmp3.setZero();
    J8.setZero();
    tmp << 0,1,0,0,
           0,0,1,0,
           0,0,0,1;

    tmp3 <<  1,0,0,0,
             0,-1,0,0,
             0,0,-1,0,
             0,0,0,-1;

    J8.topRows(4) = 2*haminus4(x.P().conj())*J.topRows(4);
    J8.bottomRows(4) = 2*(hamiplus4(x.D())*tmp3*J.topRows(4) + haminus4(x.P().conj())* J.bottomRows(4));
    tmp2.topRows(3) = tmp*J8.topRows(4);
    tmp2.bottomRows(3) = tmp*J8.bottomRows(4);

    J6.topRows(3)= tmp2.topRows(3);
    J6.bottomRows(3)= 2* tmp2.bottomRows(3);
//    J6.topRows(3) = this->getTranslationJacobian().bottomRows(3);
//    J6.bottomRows(3).setZero();
//    J6.bottomRows(1).setOnes();
    return J6;
}

MatrixXd Franka::GetJointConstraints() {
    // https://frankaemika.github.io/docs/control_parameters.html
    MatrixXd cons(2,7);
    cons << 2.8973,	1.7628,	2.8973 ,-0.0698, 2.8973, 3.7525, 2.8973, //min rad
           -2.8973,-1.7628 ,-2.8973 ,-3.0718,-2.8973 ,-0.0175 ,-2.8973; // max rad
    return cons;
}

MatrixXd Franka::GetRandomJointConfig(int n){
    MatrixXd joints(n,7);
    MatrixXd jointConstraints = GetJointConstraints();

    random_device rd;  
    mt19937 gen(rd());
    for (int j = 0; j < 7; ++j) {
        uniform_real_distribution<> dis(jointConstraints(0,j), jointConstraints(1,j));
        for (int i = 0; i < n; ++i) {
            joints(i,j) = dis(gen);
        }
    }

    return joints;

}

MatrixXd Franka::GetVelocityConstraints() {
    // https://frankaemika.github.io/docs/control_parameters.html
    MatrixXd cons(1,7);
    cons << 2.1750 ,2.1750 ,2.1750, 2.1750,2.6100 ,2.6100, 2.6100; //rad/s
    return cons;
}

DQ_VrepInterface Franka::getVREPInterface() {
    return this->m_vi;
}

bool Franka::usingVREP() {
    return m_useVREP;
}
