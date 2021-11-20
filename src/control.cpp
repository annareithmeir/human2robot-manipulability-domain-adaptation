#include <control.h>
#include <sys/stat.h>

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
void controlManipulabilitiesRHumanArm(Franka &robot, string exp, string proband, int nPoints, bool mainTask, MatrixXd &ratios){
    int num = 30; //number of random samples for interpolation data
    MatrixXd xdTmp(nPoints,4);
    MatrixXd xhatTmp(nPoints,9);
    MatrixXd xd(3, nPoints);
    MatrixXd xHat(nPoints,9);
    deb(exp)
    deb(proband)
    loadCSV("/home/nnrthmr/CLionProjects/ma_thesis/data/results/rhuman/" + exp + "/" + proband + "/xd.csv", &xd);
    loadCSV("/home/nnrthmr/CLionProjects/ma_thesis/data/results/rhuman/" + exp + "/" + proband + "/xhat.csv", &xHat);

    if (mkdir(("/home/nnrthmr/CLionProjects/ma_thesis/data/tracking/rhuman/"+exp).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1)
    {
        if( errno == EEXIST ) {
        } else {
            throw std::runtime_error( strerror(errno) );
        }
    }
    if (mkdir(("/home/nnrthmr/CLionProjects/ma_thesis/data/tracking/rhuman/"+exp+"/"+proband).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1)
    {
        if( errno == EEXIST ) {
        } else {
            throw std::runtime_error( strerror(errno) );
        }
    }

    vector<MatrixXd> mLoop;
    vector<double> eLoop;

    VectorXd dx(3);
    VectorXd x0(3);
    x0 = robot.getCurrentPosition();
    dx = xd.col(0) - x0;

    MatrixXd Mcurr;
    MatrixXd manips(xd.cols(), 9);
    MatrixXd errMatrix(xd.cols(),1);
    errMatrix.setZero();
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
        errMatrix(i,0)=(MDesired.pow(-0.5)*Mcurr*MDesired.pow(-0.5)).log().norm();
        Mcurr.resize(1,9);
        manips.row(i) = Mcurr;

        MatrixXd eLoopMat(1, eLoop.size());
        for(int i=0;i<eLoop.size();i++){
            eLoopMat(0,i)=eLoop[i];
        }

//        writeCSV(mLoop, "/home/nnrthmr/CLionProjects/ma_thesis/data/tracking/rhuman/"+exp+"/"+proband+"/loopManipulabilities.csv");
//        writeCSV(eLoopMat, "/home/nnrthmr/CLionProjects/ma_thesis/data/tracking/rhuman/"+exp+"/"+proband+"/loopErrors.csv");
    }
    deb("done");

    if(robot.usingVREP()) robot.stopSimulation();
    writeCSV(errMatrix, "/home/nnrthmr/CLionProjects/ma_thesis/data/tracking/rhuman/" + exp + "/" + proband +
                        "/errorManipulabilities.csv");
    writeCSV(manips, "/home/nnrthmr/CLionProjects/ma_thesis/data/tracking/rhuman/" + exp + "/" + proband +
                     "/controlledManipulabilities.csv");
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
    float km0 = 0.03;
    float km = km0;
    uint8_t scaledFlag =0;

    int niter=10;
    float dt=1e-2;
    float err=1000;
    MatrixXd J, M, JFull, JmT, MDiff, pinv, dqt1;
    VectorXd qt;
    robot.setJointPositions(PosInit.row(0));

    MatrixXd Jgeo(6,7);
    DQ_SerialManipulator m_robot = robot.getKinematicsDQ();

    MatrixXd MDesired(3,3);
    MDesired.setIdentity();

//    while(err>0.05){
    for(int i=0;i<niter; i++){
        qt=robot.getCurrentJointPositions();
        J = robot.getTranslationJacobian(qt).bottomRows(3);
        JFull = robot.getPoseJacobian(qt);

        Jgeo = robot.buildGeometricJacobian(JFull, qt);
        M=Jgeo.topRows(3)*Jgeo.topRows(3).transpose();
        if(i==0) cout<<M<<endl;
        JmT = robot.ComputeManipulabilityJacobian(Jgeo); // Checked!
//        JmT.bottomRows(3).setZero();

        MDiff = logMap(MDesired, M); // Checked! Like in MATLAB
        pinv = JmT.completeOrthogonalDecomposition().pseudoInverse(); // Checked!
        dqt1 = pinv * km * symmat2Vec(MDiff).transpose(); //Checked!

        robot.setJointPositions(qt + dqt1*dt);

        err=(MDesired.pow(-0.5)*M*MDesired.pow(-0.5)).log().norm();

        if(err<1.5 && scaledFlag==0) {
            // singular values
            MDesired=MDesired* getSingularValues(M).minCoeff(); // TODO min makes sense?

            //eig values like page 1 in paper
            MDesired = MDesired * getLengthOfPrincipalAxis(M);
            scaledFlag=1;
        }

        mLoop.push_back(M);
        eLoop.push_back(err);

//        km = km0* err;

        std::cout << "Tracking error: " <<err << "  new gain: " <<km<< std::endl; //norm returns Frobenius norm if Matrices
    }
    std::cout << "====================================="<< std::endl;
    cout<<M<<endl;

    DQ xFinal       = m_robot.fkm(qt).translation();
    finalPos.push_back(vec3(xFinal));
    finalM.push_back(M);
//    cout<<getSingularValues(M)<<endl;
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
