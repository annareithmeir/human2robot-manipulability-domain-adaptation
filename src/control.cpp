#include <control.h>

#define dimensions 3


// Checked for 2d!
MatrixXd manipulabilityTrackingMainTask(Franka &robot, const MatrixXd& MDesired, vector<MatrixXd> &mLoop, vector<double> &eLoop) {
    float km0 = 0.4; // 0.4 good
    float km = km0;

    int niter=1;
    float dt=1e-2;
    float err=1000;
    DQ x;
    MatrixXd J, M, JFull, JmT, MDiff, pinv, dqt1;
    MatrixXd Jgeo(6,7);
    DQ_SerialManipulator m_robot = robot.getKinematicsDQ();
    VectorXd qt= robot.getCurrentJointPositions();

    if(robot.usingVREP())  DQ_VrepInterface m_vi = robot.getVREPInterface();

//    while(err>0.05){
    for(int i=0;i<niter; i++){
        if(dimensions==2){
            qt  = robot.getCurrentJointPositions();
            J = robot.getTranslationJacobian();
            JFull = robot.getPoseJacobian();
            M = JFull*JFull.transpose();

            JmT = robot.ComputeManipulabilityJacobian(JFull); // Checked!
            MDiff = logMap(MDesired, M); // Checked!
            pinv = JmT.completeOrthogonalDecomposition().pseudoInverse(); // Checked!
            dqt1 = pinv * km * symmat2Vec(MDiff).transpose(); //Checked!

            robot.setJointPositions(qt + dqt1*dt); // Checked!
            this_thread::sleep_for(std::chrono::milliseconds(10));

            err=(MDesired.pow(-0.5)*M*MDesired.pow(-0.5)).log().norm();

            std::cout << "Tracking error: " <<err<< std::endl; //norm returns Frobenius norm if Matrices
            std::cout << "====================================="<< std::endl;
        }
        else{
            qt  = robot.getCurrentJointPositions();
            J = robot.getTranslationJacobian().bottomRows(3);
            JFull = robot.getPoseJacobian();
//            M=JFull*JFull.transpose();

            Jgeo = robot.buildGeometricJacobian(JFull, qt);
            deb(Jgeo)
            M=Jgeo.topRows(3)*Jgeo.topRows(3).transpose();
            deb(M)
//            M=J*J.transpose();
            JmT = robot.ComputeManipulabilityJacobian(Jgeo); // Checked!
            deb(JmT)
            JmT.bottomRows(3).setZero();
//            deb(JmT)

            MDiff = logMap(MDesired, M); // Checked! Like in MATLAB
            deb(MDiff)
            pinv = JmT.completeOrthogonalDecomposition().pseudoInverse(); // Checked!
            deb(pinv)
            dqt1 = pinv * km * symmat2Vec(MDiff).transpose(); //Checked!
            deb(symmat2Vec(MDiff).transpose())
            deb(dqt1)

            robot.setJointPositions(qt + dqt1*dt); // Checked!
            std::this_thread::sleep_for(std::chrono::milliseconds(10));

            err=(MDesired.pow(-0.5)*M*MDesired.pow(-0.5)).log().norm();

            mLoop.push_back(M);
            eLoop.push_back(err);

            km = km0* err;

            std::cout << "Tracking error: " <<err << "  new gain: " <<km<< std::endl; //norm returns Frobenius norm if Matrices
        }
    }
    std::cout << "====================================="<< std::endl;
    return M;
}

// Not checked yet
MatrixXd manipulabilityTrackingNullspace(Franka &robot, const MatrixXd& MDesired, vector<MatrixXd> &mLoop, vector<double> &eLoop) {
    float km0 = 0.4; // 0.4 good
    float km = km0;
    float knp = 20;
    float knd = 2;

    int niter=1;
    float dt=1e-2;
    float err=1000;
    DQ x;
    MatrixXd J, M, JFull, JmT, MDiff, pinv, dqt1, dqns;
    MatrixXd Jgeo(6,7);
    DQ_SerialManipulator m_robot = robot.getKinematicsDQ();
    VectorXd qt= robot.getCurrentJointPositions();
    VectorXd qh= robot.getCurrentJointPositions();
    VectorXd dqt = VectorXd(7).setZero();
    VectorXd desiredManipVelocity = VectorXd(3).setZero();

    double threshold = 1e-2;
    double minEigJmT;
    MatrixXd eyeNDoF(7,7);
    eyeNDoF.setIdentity();
    MatrixXd dMeP(1, 6);
    dMeP.setZero();

    bool keepDesiredJointAngle = true;
    MatrixXd Wq(7,7);
    Wq.setIdentity(); //Which joint angles should stay fixed

    if(robot.usingVREP())  DQ_VrepInterface m_vi = robot.getVREPInterface();

//    while(err>0.05){
    for(int i=0;i<niter; i++){
        qt  = robot.getCurrentJointPositions();
        J = robot.getTranslationJacobian().bottomRows(3);
        JFull = robot.getPoseJacobian();
//            M=JFull*JFull.transpose();

        Jgeo = robot.buildGeometricJacobian(JFull, qt);
        deb(Jgeo)
        M=Jgeo.topRows(3)*Jgeo.topRows(3).transpose();
        deb(M)
//            M=J*J.transpose();
        JmT = robot.ComputeManipulabilityJacobian(Jgeo); // Checked!
        deb(JmT)
        JmT.bottomRows(3).setZero();
//            deb(JmT)

        MDiff = logMap(MDesired, M); // Checked! Like in MATLAB
        deb(MDiff)

//        //singular avoidance
//        JacobiSVD<MatrixXf> svd(JmT, ComputeThinU | ComputeThinV);
//        minEigJmT = svd.singularValues().minCoeff();
//        if (minEigJmT < threshold) pinv = (JmT.transpose()*JmT + threshold * eyeNDoF).inverse() * JmT.transpose();
//        else pinv = JmT.completeOrthogonalDecomposition().pseudoInverse(); // Checked!
//        dMeP = symmat2Vec(desiredManipVelocity) + km* symmat2Vec(MDiff);
//        dqt1 = pinv *dMeP;
//
//        if (keepDesiredJointAngle){
//            dqns = (eyeNDoF - pinv * JmT) * Wq * (knp * (qh-qt) - knd*dqt);
//            dqt = dqt1 +dqns;
//        }
//        else{
//            dqt = dqt1;
//        }

        qt = qt + dqt*dt;



        pinv = JmT.completeOrthogonalDecomposition().pseudoInverse(); // Checked!
        deb(pinv)
        dqt1 = pinv * km * symmat2Vec(MDiff).transpose(); //Checked!
        deb(symmat2Vec(MDiff).transpose())
        deb(dqt1)

        robot.setJointPositions(qt + dqt1*dt); // Checked!
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        err=(MDesired.pow(-0.5)*M*MDesired.pow(-0.5)).log().norm();

        mLoop.push_back(M);
        eLoop.push_back(err);

        km = km0* err;

        std::cout << "Tracking error: " <<err << "  new gain: " <<km<< std::endl; //norm returns Frobenius norm if Matrices

    }
    std::cout << "====================================="<< std::endl;
    return M;
}

MatrixXd manipulabilityTrackingSecondaryTask(Franka robot, const MatrixXd& XDesired, const MatrixXd& DXDesired, const MatrixXd& MDesired) {
    float dt=1e-2;
    int niter=200;
    int kp=3; //gain for position control
    int km=0; // gain for manip in null space
    int nDoF = 4; //robot dofs

    DQ x;
    double errPos=1000, errManip=1000;
    MatrixXd J, M, qt, JmT, DxR, pinv, dqt1, MDiff, pinv2, mnsCommand, dqns, JFull, xPos;
    if(dimensions==2) xPos = MatrixXd(2,1);
    else xPos=MatrixXd(3,1);
    DQ_SerialManipulator m_robot = robot.getKinematicsDQ();

//    DQ_VrepInterface m_vi = robot.getVREPInterface();

//    while(errPos>0.02) {
//    while(errPos>0.02 || errManip>0.6) {
    for(int i=0;i<niter; i++) {
        if(dimensions==2){
            qt = robot.getCurrentJointPositions();
            x = m_robot.fkm(qt);
            cout<<"x DQ: "<<x.vec8();
            xPos << x.translation().vec3()[0], x.translation().vec3()[1];

            J = robot.getPoseJacobian();
            M = J * J.transpose();

            JmT = robot.ComputeManipulabilityJacobian(J); // Checked!
            deb(JmT)

            // Compute joint velocities
            DxR = DXDesired + kp*(XDesired-xPos);
            pinv = J.completeOrthogonalDecomposition().pseudoInverse();
            dqt1 = pinv*DxR;

            // Compute null space joint velocities
            MDiff = logMap(MDesired, M); // Checked!
            pinv2 = JmT.completeOrthogonalDecomposition().pseudoInverse();
            mnsCommand = pinv2 * symmat2Vec(MDiff);
            dqns = (MatrixXd::Identity(nDoF, nDoF) - pinv*J)*km*mnsCommand; // Redundancy resolution

            robot.setJointPositions(qt + (dqt1+dqns)*dt);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));

            std::cout << "Position/Manipulability tracking error: " <<(XDesired-xPos).norm()<<"   "<<(MDesired.pow(-0.5)*M*MDesired.pow(-0.5)).log().norm()<< std::endl;
//            std::cout << "Manipulability tracking error: " <<(MDesired.pow(-0.5)*M*MDesired.pow(-0.5)).log().norm()<< std::endl; //norm returns Frobenius norm if Matrices
            std::cout << "====================================="<< std::endl;
        }
        else{
            qt = robot.getCurrentJointPositions();
            x = m_robot.fkm(qt);
            xPos << x.translation().vec3()[0], x.translation().vec3()[1], x.translation().vec3()[2];

            JFull = robot.getPoseJacobian();
            J = robot.getTranslationJacobian().bottomRows(3);
            M=J*J.transpose();


            JmT = robot.ComputeManipulabilityJacobian(JFull); // Checked!

            // Compute joint velocities
            DxR = DXDesired + kp*(XDesired-xPos);
            pinv = J.completeOrthogonalDecomposition().pseudoInverse();
            dqt1 = pinv*DxR;

            // Compute null space joint velocities
            MDiff = logMap(MDesired, M); // Checked!
            pinv2 = JmT.completeOrthogonalDecomposition().pseudoInverse();
            mnsCommand = pinv2 * symmat2Vec(MDiff).transpose();
            dqns = (MatrixXd::Identity(robot.m_dof, robot.m_dof) - pinv*J)*km*mnsCommand; // Redundancy resolution
            robot.setJointPositions(qt + (dqt1+dqns)*dt);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));

            errPos=(XDesired-xPos).norm();
            errManip=(MDesired.pow(-0.5)*M*MDesired.pow(-0.5)).log().norm();

            std::cout << "Position/Manipulability tracking error: " <<errPos<<"   "<<errManip<< std::endl;
        }
    }
    std::cout << "====================================="<< std::endl;
    return M;
}

/**
 *  Control only manipulabilities of given human arm movement
 */
void controlManipulabilitiesRHumanArm(Franka &robot, MatrixXd &xd, MatrixXd &xHat, int nPoints, bool mainTask, MatrixXd &ratios, MatrixXd &errors, MatrixXd &controlledManips){
    int num = 30; //number of random samples for interpolation data
//    MatrixXd xdTmp(nPoints,4);
//    MatrixXd xhatTmp(nPoints,9);
//    MatrixXd xd(3, nPoints);
//    MatrixXd xHat(nPoints,9);

//    loadCSV("/home/nnrthmr/CLionProjects/ma_thesis/data/learning/rhuman/" + exp + "/" + proband + "/xd.csv", &xd);
//    loadCSV("/home/nnrthmr/CLionProjects/ma_thesis/data/learning/rhuman/" + exp + "/" + proband + "/xhat.csv", &xHat);

//    if (mkdir(("/home/nnrthmr/CLionProjects/ma_thesis/data/tracking/rhuman/"+exp).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1)
//    {
//        if( errno == EEXIST ) {
//        } else {
//            throw std::runtime_error( strerror(errno) );
//        }
//    }
//    if (mkdir(("/home/nnrthmr/CLionProjects/ma_thesis/data/tracking/rhuman/"+exp+"/"+proband).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1)
//    {
//        if( errno == EEXIST ) {
//        } else {
//            throw std::runtime_error( strerror(errno) );
//        }
//    }

    vector<MatrixXd> mLoop;
    vector<double> eLoop;

    VectorXd dx(3);
    VectorXd x0(3);
    x0 = robot.getCurrentPosition();
    dx = xd.col(0) - x0;

    MatrixXd Mcurr;
//    MatrixXd manips(xd.cols(), 9);
//    MatrixXd errMatrix(xd.cols(),1);
//    errMatrix.setZero();
    if(robot.usingVREP())  robot.startSimulation();

    for(int i=0;i<1;i++){
//    for(int i=0;i<100;i++){
//    for(int i=0;i<xd.cols();i++){
        if(i>0) dx = xd.col(i) - xd.col(i-1);
        MatrixXd MDesired = xHat.row(i);
        MDesired.resize(3,3);
        MDesired = MDesired * ratios(0,i);

        if(mainTask) Mcurr = manipulabilityTrackingSecondaryTask(robot, xd.col(i), dx, MDesired);
        else Mcurr=manipulabilityTrackingMainTask(robot, MDesired, mLoop, eLoop);
        errors(i,0)=(MDesired.pow(-0.5)*Mcurr*MDesired.pow(-0.5)).log().norm();
        Mcurr.resize(1,9);
        controlledManips.row(i) = Mcurr;

        MatrixXd eLoopMat(1, eLoop.size());
        for(int i=0;i<eLoop.size();i++){
            eLoopMat(0,i)=eLoop[i];
        }

//        writeCSV(mLoop, "/home/nnrthmr/CLionProjects/ma_thesis/data/tracking/rhuman/"+exp+"/"+proband+"/loopManipulabilities.csv");
//        writeCSV(eLoopMat, "/home/nnrthmr/CLionProjects/ma_thesis/data/tracking/rhuman/"+exp+"/"+proband+"/loopErrors.csv");
    }
    deb("done");

    if(robot.usingVREP()) robot.stopSimulation();
}

/**
 *
 * @param PosInit initial posotion to start from
 * @param MDesired manipulability we control for
 * @param finalM resulting manipulability
 * @param finalPos resulting EE position
 * @param mLoop vector of resulting manipulabilities throughout loop
 * @param eLoop vector of errors throughout loop
 */
void unitShpereTrackingMainTask(Franka robot, const MatrixXd& PosInit, vector<MatrixXd> &finalM, vector<MatrixXd> &finalPos, vector<MatrixXd> &mLoop, vector<double> &eLoop) {
    float km = 15;
    double singularRegion = 0.1;
    double dampingFactorMax= 0.0001;

    uint8_t scaledFlag =0;

    int niter=1000;
    float dt=1e-2;
    float err=1000;
    MatrixXd J, M, JFull, JmT, MDiff, pinv, dqt1, errMatrix(1000,1);
    VectorXd qt;
    robot.setJointPositions(PosInit.row(0));

    MatrixXd Jgeo(6,7);
    DQ_SerialManipulator m_robot = robot.getKinematicsDQ();

    MatrixXd MDesired(3,3);
    MDesired.setIdentity();

    MatrixXd S, U,Um;
    int rank;
    double minSingVal, dampingFactor;

//    while(err>0.05){
    for(int i=0;i<niter; i++){
        qt=robot.getCurrentJointPositions();
        JFull = robot.getPoseJacobian(qt);

        Jgeo = robot.buildGeometricJacobian(JFull, qt);
        M=Jgeo.bottomRows(3)*Jgeo.bottomRows(3).transpose();
        if(i==0) cout<<M<<endl;

        JmT = robot.ComputeManipulabilityJacobianLower(Jgeo); // taskvar 4:6

        MDiff = logMap(MDesired, M); // Checked! Like in MATLAB
        err=symmat2Vec(MDiff).norm();

        JacobiSVD<MatrixXd> svd(JmT, ComputeThinU | ComputeThinV);

        S = svd.singularValues().real();
        U=svd.matrixU();
        rank= svd.rank();
        minSingVal = S(rank-1, 0);
//        deb(U)
//        deb(S)
//        deb(minSingVal)
//        deb(rank)

        MatrixXd Umtmp(6,rank);
        Umtmp.setZero();

        int cnt=0;
        if(minSingVal<=singularRegion){
            for(int ii=rank-1; ii>=0;--ii){
                if(S(ii,0)<=singularRegion){
                    Umtmp.col(ii) = U.col(ii);
                    cnt++;
                }
            }
            dampingFactor = (1-pow(minSingVal/singularRegion,2))*dampingFactorMax;
        }
        else{
            dampingFactor=0;
        }

        Um = Umtmp.rightCols(cnt);
//        deb(Um)
//        deb(cnt)

//        pinv = JmT.completeOrthogonalDecomposition().pseudoInverse(); // Checked!
//        dqt1 = pinv * km * symmat2Vec(MDiff).transpose(); //Checked!

//        pinv = JmT.transpose()*(JmT*JmT.transpose() + 0.1*MatrixXd::Identity(6,6)).completeOrthogonalDecomposition().pseudoInverse(); // Checked!
        pinv = JmT.transpose()*(JmT*JmT.transpose() + dampingFactor*Um*Um.transpose()).completeOrthogonalDecomposition().pseudoInverse(); // Checked!
        dqt1 = pinv * (km/err) * symmat2Vec(MDiff).transpose(); //Checked!

        robot.setJointPositions(qt + dqt1*dt);

//        err=(MDesired.pow(-0.5)*M*MDesired.pow(-0.5)).log().norm();

//        if(err<0.2 && scaledFlag==0) {
//            // singular values
////            MDesired=MDesired* getSingularValues(M).minCoeff(); // TODO min makes sense?
//
//            //eig values like page 1 in paper
//            MDesired = MDesired * getLengthOfPrincipalAxis(M);
//            scaledFlag=1;
//        }

        mLoop.push_back(M);
        eLoop.push_back(err);
        errMatrix(i, 0)=err;

        std::cout << "Tracking error: " <<err << std::endl; //norm returns Frobenius norm if Matrices
    }
    std::cout << "====================================="<< std::endl;
    cout<<M<<endl;

    DQ xFinal       = m_robot.fkm(qt).translation();
    finalPos.push_back(vec3(xFinal));
    finalM.push_back(M);

    writeCSV(errMatrix, "/home/nnrthmr/errTest.csv");
}

void precomputeScalingRatios(Franka &robot, MatrixXd &xd, MatrixXd &ratios){
    assert(ratios.cols()==xd.cols());
    int num = 30;
    getScalingRatiosAtPoints(num, xd, ratios);
}

/**
 * Calibration process to calculate the scale distribution when controling for unit sphere.
 * @param positions 3xnum matrix, colwise positions of calibration process
 * @param scales 1xnum scales from process
 */
void calibrationProcessRobot(Franka robot, MatrixXd &positions, MatrixXd &scales){
    assert(positions.cols() == scales.cols());
    assert(positions.rows()==3);
    assert(scales.rows()==1);

    int num = positions.cols();
    cout<< "Calibration process based on "<<num<<" random configurations of the robot..."<<endl;

    vector<double> eLoop;
    vector<MatrixXd> mLoop, finalM, finalPos;

    MatrixXd randomJoints = robot.GetRandomJointConfig(num); // num x 7, colwise
    MatrixXd MDesired;
    MDesired.setIdentity();

    for(int i=0;i<num; ++i){
        // finalM and finalPos: num elements, mLoop and eLoop: num*iter elements
        unitShpereTrackingMainTask(robot, randomJoints.row(i),  finalM, finalPos, mLoop, eLoop);
    }
    assert(finalM.size()==num);
    assert(finalPos.size()==num);


    for(int i=0; i< num;++i){
        scales(0,i)=getSingularValues(finalM[i]).minCoeff(); //TODO makes sense?
        positions.col(i) = finalPos[i];
    }
}

/**
 * Calibration process for human arm to calculate the scale distribution when controling for unit sphere.
 * @param positions 3xnum matrix, colwise positions of calibration process
 * @param scales 1xnum scales from process
 */
void calibrationProcessHuman(MatrixXd &positions, MatrixXd &scales, double shoulderHeight){
    assert(positions.cols() == scales.cols());
    assert(positions.rows()==3);
    assert(scales.rows()==1);

    int num = positions.cols();
    cout<< "Calibration process based on "<<num<<" random configurations of the human..."<<endl;

    unique_ptr<MATLABEngine> matlabPtr = startMATLAB();
    matlab::data::ArrayFactory factory;

    matlab::data::TypedArray<double>  args_shoulderHeight = factory.createScalar<double>(shoulderHeight);
    matlab::data::TypedArray<int>  args_num = factory.createScalar<int>(num);

    matlabPtr->setVariable(u"shoulderHeight_m", std::move(args_shoulderHeight));
    matlabPtr->setVariable(u"num_m", std::move(args_num));

    matlabPtr->eval(u"[scales, positions]=calibrationHumanArm(num_m, shoulderHeight_m);");
    matlab::data::TypedArray<double> sc = matlabPtr->getVariable(u"scales");
    matlab::data::TypedArray<double> pos = matlabPtr->getVariable(u"positions");

    for(int i=0;i<num;++i){
        positions(0,i)=pos[i][0];
        positions(1,i)=pos[i][1];
        positions(2,i)=pos[i][2];
        scales(0,i) = sc[0][i];
    }
}
