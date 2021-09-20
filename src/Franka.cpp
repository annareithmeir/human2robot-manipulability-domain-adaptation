#include <chrono>
#include <thread>
#include "Franka.h"

#define KM 3
#define DT 1e-2
#define NITER 65

// Checked!
vector<MatrixXd> LogMap(vector<MatrixXd> X, MatrixXd S) {
    vector<MatrixXd> U;
    for (int i = 0; i < X.size(); i++) {
        MatrixXd tmp = (S.inverse()) * X[i]; //A\B in MATLAB is a^-1 * B
        EigenSolver<MatrixXd> es(tmp);
        MatrixXd D = es.eigenvalues().real().asDiagonal();
        MatrixXd V = es.eigenvectors().real();
        MatrixXd tmp2 = D.diagonal().array().log().matrix().asDiagonal().toDenseMatrix();
        U.push_back(S * V * tmp2 * V.inverse());
    }
    return U;
}

//// Checked!
//MatrixXd LogMap(MatrixXd X, MatrixXd S) {
//    MatrixXd U;
//    MatrixXd tmp = (S.inverse()) * X; //A\B in MATLAB is a^-1 * B
//    EigenSolver<MatrixXd> es(tmp);
//    MatrixXd D = es.eigenvalues().real().asDiagonal();
//    MatrixXd V = es.eigenvectors().real();
//    MatrixXd tmp2 = D.diagonal().array().log().matrix().asDiagonal().toDenseMatrix();
//    U = S * V * tmp2 * V.inverse();
//    return U;
//}

// Checked!
void CumulativeSum(const VectorXd &input, VectorXd &result) {
    result(0) = input[0];
    for (int i = 1; i < input.size(); i++) {
        result(i) = result(i - 1) + input(i);
    }
}

// Checked!
vector<MatrixXd> Vec2Symmat(MatrixXd vec) {
    vector<MatrixXd> MVector;
    int d = vec.rows();
    int N = vec.cols();
    int D = (-1 + sqrt(1 + 8 * d)) / (double) 2;
    int row;
    for (int n = 0; n < N; n++) { //colwise
        MatrixXd vn = vec.col(n).transpose();
        MatrixXd Mn = vn.row(0).leftCols(D).asDiagonal();
        VectorXd id(D);
        CumulativeSum(VectorXd::LinSpaced(D, D, 1), id);
        MatrixXd tmp1(Mn.rows(), Mn.cols());
        MatrixXd tmp2(Mn.rows(), Mn.cols());
        for (int i = 1; i < D; i++) {
            tmp1.setZero();
            row = 0;
            for (int k = i; k < id(i) - id(i - 1) + i; k++) {
                tmp1(row, k) = vn(0, id(i - 1) + row) * (1 / sqrt(2));
                row++;
            }
            tmp2.setZero();
            row = 0;
            for (int k = i; k < id(i) - id(i - 1) + i; k++) {
                tmp2(k, row) = vn(0, id(i - 1) + row) * (1 / sqrt(2));
                row++;
            }
            Mn = Mn + tmp1 + tmp2;
        }
        MVector.push_back(Mn);
    }
    return MVector;
}

// Checked!
MatrixXd Symmat2Vec(MatrixXd mat) {
    int N = mat.rows();
    vector<double> v;
    VectorXd dia = mat.diagonal();
    for (int x = 0; x < dia.size(); x++) {
        v.push_back(dia(x));
    }
    int row, col;
    for (int n = 1; n < N; n++) {
        row = 0;
        col = n;
        for (int ni = n; ni < N; ni++) {
            v.push_back(sqrt(2) * mat(row, col));
            row++;
            col++;
        }
    }
    MatrixXd vm(1, v.size());
    for (int x = 0; x < v.size(); x++) {
        vm(0, x) = v[x]; //one row
    }
    return vm;
}

//// Checked!
//vector<MatrixXd> LogmapVec(MatrixXd x, MatrixXd s) {
//    vector<MatrixXd> X = Vec2Symmat(x);
//    vector<MatrixXd> S = Vec2Symmat(s);
//    vector<MatrixXd> U = LogMap(X, S[0]); //Vec2Symmat gives back vector of size 1 here
//    vector<MatrixXd> u = Symmat2Vec(U);
//    return u;
//}

// Checked!
//vector<MatrixXd> Symmat2Vec(vector<MatrixXd> mat_vec) {
//    int N = mat_vec.size();
//    vector<MatrixXd> vec;
//    for (int i = 0; i < N; i++) {
//        MatrixXd vn = Symmat2Vec(mat_vec[i]);
//        vec.push_back(vn);
//    }
//    return vec;
//}

Franka::Franka(){
    this->m_kinematics= MatrixXd(5,7);

    const double pi2 = pi/2.0;
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

void Franka::moveToQGoal(VectorXd q_goal){

    DQ_SerialManipulator m_robot = this->getKinematicsDQ();
    std::cout << "Starting V-REP simulation..." << std::endl;
    m_vi.start_simulation();
    DQ xd = m_robot.fkm(q_goal);

    VectorXd q;
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
        DQ x       = m_robot.fkm(q);
        // Compute error in the Task-Space
        e          = vec8(x-xd);
        // Obtain the current analytical Jacobian (Dim: 8 * n)
        MatrixXd J = m_robot.pose_jacobian(q);
        // Kinematic Control Law
        VectorXd u = -0.01 * pinv(J) * e;
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

// Checked for 2d!
void Franka::ManipulabilityTrackingMainTask(MatrixXd MDesired) {
    DQ x;
    MatrixXd J, M, qt;
    DQ_SerialManipulator m_robot = this->getKinematicsDQ();
    x = m_robot.fkm(qt);

    for(int i=0;i<NITER; i++){
        qt  = m_vi.get_joint_positions(m_jointNames);
        J = this->getJacobian();
        M = J*J.transpose();

        MatrixXd JmT = ComputeManipulabilityJacobian(J); // Checked!
//        MatrixXd MDiff = LogMap(MDesired, M); // Checked!
//        MatrixXd pinv = jmt.completeOrthogonalDecomposition().pseudoInverse(); // Checked!
//        MatrixXd dqt1 = pinv*3*model2.Symmat2Vec(mdiff).transpose(); //Checked!
//
//        m_vi.set_joint_positions(m_jointNames,qt + dqt1*DT); // Checked!
//        std::this_thread::sleep_for(std::chrono::milliseconds(10));
//
//        std::cout << "Tracking error: " <<(MDesired.pow(-0.5)*M*MDesired.pow(-0.5)).log().norm()<< std::endl; //norm returns Frobenius norm if Matrices
//        std::cout << "====================================="<< std::endl;
    }
}

void Franka::ManipulabilityTrackingSecondaryTask(MatrixXd MDesired) {
    //TODO
}

std::vector<MatrixXd> Franka::ComputeTensorMatrixProduct(std::vector<MatrixXd> T, MatrixXd U, int mode){
//    MatrixXd sizeTens = MatrixXd::Ones(1, mode);
    MatrixXd sizeTens(1,3);
    sizeTens<< T[0].rows(), T[0].cols(), T.size();
    int N = 3;

//    cout<< " SizeTens:" <<sizeTens<<endl;
//    cout<< " N:" <<N<<endl;

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

//    cout<< " bits:" <<bits<<endl;
//    cout<< " modec:" <<modeC<<endl;

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

//    for(int x=0;x<perm.size();x++) {
//        cout << " perm:" << perm[x] << endl;
//        cout << " iperm:" << iperm[x] << endl;
//    }
//
//    for(int x=0;x<3;x++) {
//        cout << " sizetensperm:" << sizeTensPerm(0,x) << endl;
//    }


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

//    for(int z=0;z<S2.size();z++)
//        cout<<"S2\n"<<S2[z]<<endl;
//    cout<<iperm.size()<<endl;
    for(int i=0;i<perm.size();i++){
        iperm[perm[i]] = i;
    }

    if(mode!=1){
        assert(iperm[0]==1 && iperm[1]==0 && iperm[2]==2);
        for(int i=0;i<S2.size();i++){
            MatrixXd tmp4 = S2[i].transpose();
            S2[i]=tmp4;
//            cout<<S2[i]<<"\n\n"<<endl;
        }
    }

//    for(int i=0;i<S2.size();i++){
//        cout<<S2[i]<<"\n\n"<<endl;
//    }

    return S2;
}

// Checked!
vector<MatrixXd> Franka::ComputeJointDerivative(MatrixXd J){
    std::vector<MatrixXd> JGradVec;
    for(int i=0;i<J.cols();i++){
        MatrixXd tmp(J.rows(), J.cols());
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

//    for(int x=0;x<JGradVec.size();x++) {
//        cout << JGradVec[x] << endl;
//        cout << "\n" << endl;
//    }

    return JGradVec;
}

// Checked!
MatrixXd Franka::ComputeManipulabilityJacobian(MatrixXd J){
    MatrixXd W = MatrixXd::Identity(J.cols(), J.cols());
    std::vector<MatrixXd> JGradVec;
    JGradVec = ComputeJointDerivative(J);
    std::vector<MatrixXd> Jm;
    MatrixXd U= J*(W*W.transpose());
    vector<MatrixXd> tmp1= ComputeTensorMatrixProduct(JGradVec, U, 2);

    for(int i=0;i<JGradVec.size();i++) {
        MatrixXd tmp = JGradVec[i].transpose();
        JGradVec[i] = tmp;
    }

    vector<MatrixXd> tmp2 = ComputeTensorMatrixProduct(JGradVec, U, 1);
    for(int i=0;i<tmp1.size();i++) {
        MatrixXd tmp3 = tmp1[i] + tmp2[i];
        Jm.push_back(tmp3);
    }

    MatrixXd JmRed(3, Jm.size());
    JmRed.setZero();
    for(int i=0;i<Jm.size();i++)
        JmRed.col(i)=(Symmat2Vec(Jm[i].block(0,0,2,2))).transpose(); //TaskVar 1:2 in MATLAB

    return JmRed;
}
