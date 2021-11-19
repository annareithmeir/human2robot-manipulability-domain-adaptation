#include <Franka.h>
#include <GMM.h>
#include <GMM_SPD.h>
#include <control.h>
#include <learn.h>
#include <errno.h>
#include <sys/stat.h>

using namespace std;
using namespace Eigen;

//#define deb(x) cout << #x << " " << x << endl;

void transfer3d(Franka robotStudent){
    // Build GMM for trajectories in cartesian coordinates
    string cmat_path="/home/nnrthmr/CLionProjects/ma_thesis/data/demos/trajectories.csv";
    MatrixXd data(80, 4);
    loadCSVSkipFirst(cmat_path, &data);

    GMM model = GMM();
    model.InitModel(data.transpose(), 4);
    model.TrainEM();

    MatrixXd xd(3, 20);
    xd.setZero();
    std::vector<MatrixXd> expSigma; //m_dimOut x m_dimOut x m_nData

//     Build GMM for Manifold
    string mmat_path = "/home/nnrthmr/CLionProjects/ma_thesis/data/demos/translationManip3d.csv";
    MatrixXd data2(80, 10);
    MatrixXd dataVectorized(80, 7);
    loadCSVSkipFirst(mmat_path, &data2);

    for (int i = 0; i < data2.rows(); i++) {
        dataVectorized(i, 0) = data2(i,0);
        MatrixXd tmp = data2.block(i, 1, 1, 9);
        tmp.resize(3, 3);
        dataVectorized.block(i, 1, 1, 6) = symmat2Vec(tmp);
    }

    GMM_SPD model2 = GMM_SPD();
    model2.InitModel(dataVectorized, 4);
    model2.TrainEM();

    MatrixXd xHat(6, 20);
    xHat.setZero();
    std::vector<MatrixXd> expSigma2; //m_dimOut x m_dimOut x m_nData

    VectorXd q0=robotStudent.getCurrentJointPositions();
    VectorXd qt=q0;
    MatrixXd Jt, JtFull, Htmp, ManipCurr, pinvJt, dqT1, mCommandNs, dqns, Jmt, pinv;
    MatrixXd mDiff;
    VectorXd xCurr;
    vector<MatrixXd> Jmtmp;
    MatrixXd xt(3, model2.m_n);
    xt.setZero();
    MatrixXd ei= MatrixXd(q0.size(), q0.size());
    ei.setIdentity()*model2.m_regTerm;

    vector<VectorXd> robotStudentQ;
    vector<VectorXd> robotStudentPos;
    for(int t=0; t<model2.m_n; t++){

        model.GMR(xd, expSigma, t);
        model2.GMR(xHat, expSigma2, t);
        Jt = robotStudent.getTranslationJacobian(qt).bottomRows(3); //top row zeros
        JtFull = robotStudent.getPoseJacobian(qt);
        Htmp = robotStudent.getCurrentPosition(qt);
        xt.col(t) = Htmp ;
        ManipCurr = Jt*Jt.transpose();
        robotStudentQ.push_back(qt);
        robotStudentPos.push_back(xCurr);

        Jmt = robotStudent.ComputeManipulabilityJacobian(JtFull);

        //desired joint velocities
        pinvJt = (Jt.transpose() * expSigma[t] * Jt + 1e-6 * MatrixXd::Identity(7,7)).completeOrthogonalDecomposition().pseudoInverse() * (Jt.transpose() * expSigma[t]);
        dqT1 = pinvJt * (model2.m_kp*(xd.col(t)-xt.col(t)));
        mDiff = logMap(vec2Symmat(xHat.col(t))[0], ManipCurr); //only one element because colwise

        //manipulability tracking in nullspace of position command
        pinv=Jmt.completeOrthogonalDecomposition().pseudoInverse();
        mCommandNs = pinv * symmat2Vec(mDiff).transpose();
        pinv=Jt.completeOrthogonalDecomposition().pseudoInverse();
        dqns=(MatrixXd::Identity(robotStudent.m_dof,robotStudent.m_dof) - pinv*Jt) * model2.m_km * mCommandNs;
        qt = qt + ( dqT1 + dqns )* model2.m_dt;
        deb(qt);
//        robotStudent.setJoints(qt);
    }
    std::cout<<"GMR"<<std::endl;

    //Write model2.muMan, model2.sigma, xhat to files
    writeCSV(model2.m_muMan, "/home/nnrthmr/CLionProjects/ma_thesis/data/model2MuMan.csv");
    writeCSV(model2.m_sigma, "/home/nnrthmr/CLionProjects/ma_thesis/data/model2Sigma.csv");
    writeCSV(vec2Symmat(xHat), "/home/nnrthmr/CLionProjects/ma_thesis/data/xhat.csv");
}

/**
 * This function learns from the simulated 3d data from VREP and then controls the robot in VREP to follow
 * according to the learned data.
 */
void learnAndConrol(){
    Franka robot = Franka(false);
    MatrixXd xd(3, 20);
    MatrixXd xHat(6, 20);
    learn3d(xd, xHat);
    std::cout<<"Control ..."<<std::endl;

    VectorXd dx(3);
    VectorXd x0(3);
    x0 = robot.getCurrentPosition();
    dx = xd.col(0) - x0;
    deb(x0);

    MatrixXd Mcurr;
//    MatrixXd saveData(xHat.cols(), 18);
//    saveData.setZero();

//    for(int i=0;i<1;i++){
    for(int i=0;i<xHat.cols();i++){
        if(i>0) dx = xd.col(i) - xd.col(i-1);
        deb(dx);
        deb(xd.col(i));
        Mcurr = ManipulabilityTrackingSecondaryTask(robot, xd.col(i), dx, vec2Symmat(xHat.col(i))[0]);

//        Mcurr=robot.ManipulabilityTrackingMainTask(vec2Symmat(xHat.col(i))[0]);
//        saveData.block(i,0,1,3) = vec2Symmat(xHat.col(i))[0].row(0);
//        saveData.block(i,3,1,3) = vec2Symmat(xHat.col(i))[0].row(1);
//        saveData.block(i,6,1,3) = vec2Symmat(xHat.col(i))[0].row(2);
//        saveData.block(i,9,1,3) = Mcurr.row(0);
//        saveData.block(i,12,1,3) = Mcurr.row(1);
//        saveData.block(i,15,1,3) = Mcurr.row(2);
    }
    deb("done");

//    writeCSV(saveData, "/home/nnrthmr/CLionProjects/ma_thesis/data/tracking/test.csv");
}


/**
 * This function learns from the simulated 3d data from VREP and then controls the robot in VREP to follow
 * according to the learned data.
 */
void control(){
    Franka robot = Franka(false);
    MatrixXd xdTmp(80,4);
    MatrixXd xd(3, 20);
    MatrixXd xHat(20,9);
    loadCSVSkipFirst("/home/nnrthmr/CLionProjects/ma_thesis/data/demos/trajectories.csv", &xdTmp);
    xd=xdTmp.rightCols(3).topRows(20).transpose();
    loadCSV("/home/nnrthmr/CLionProjects/ma_thesis/data/xhat.csv", &xHat);

    VectorXd dx(3);
    VectorXd x0(3);
    x0 = robot.getCurrentPosition();
    dx = xd.col(0) - x0;

    vector<MatrixXd> mLoop;
    vector<double> eLoop;
    MatrixXd Mcurr;
    MatrixXd manips(xd.cols(), 9);
    MatrixXd errMatrix(xd.cols(),1);
    errMatrix.setZero();

    robot.startSimulation();

    for(int i=0;i<1;i++){
//    for(int i=0;i<xd.cols();i++){
        if(i>0) dx = xd.col(i) - xd.col(i-1);
//        MatrixXd MDesired(1,9);
//        MDesired << 0.112386331709752,	-0.292084314237333,	0.085136827901769,	-0.292084314237333,	0.892749865686052,	0.007596007158765,	0.085136827901769,	0.007596007158765,	0.698160049993724;
        MatrixXd MDesired = xHat.row(i);
        MDesired.resize(3,3);
//        Mcurr = ManipulabilityTrackingSecondaryTask(robot, xd.col(i), dx, MDesired);
        Mcurr=ManipulabilityTrackingMainTask(robot, MDesired, mLoop, eLoop);
        errMatrix(i,0)=(MDesired.pow(-0.5)*Mcurr*MDesired.pow(-0.5)).log().norm();
        Mcurr.resize(1,9);
        manips.row(i) = Mcurr;
    }
    deb("done");
    robot.stopSimulation();

    writeCSV(errMatrix, "/home/nnrthmr/CLionProjects/ma_thesis/data/tracking/errorManipulabilities.csv");
    writeCSV(manips, "/home/nnrthmr/CLionProjects/ma_thesis/data/tracking/xhat.csv");
}
/**
 *  Control only manipulabilities of given human arm movement
 */
void controlManipulabilitiesHumanArm(){
    Franka robot = Franka(false);
    MatrixXd xdTmp(1600,4);
    MatrixXd xhatTmp(1600,9);
    MatrixXd xd(3, 400);
    MatrixXd xHat(400,9);
    loadCSV("/home/nnrthmr/CLionProjects/ma_thesis/data/demos/human_arm/dummyTrajectories.csv", &xdTmp);
    xd=xdTmp.rightCols(3).topRows(400).transpose();
    xd=xd*10;
    loadCSV("/home/nnrthmr/CLionProjects/ma_thesis/data/demos/human_arm/dummyManipulabilities.csv", &xhatTmp);
    xHat = xhatTmp.topRows(400);
    xHat=xHat*10;

    VectorXd dx(3);
    VectorXd x0(3);
    x0 = robot.getCurrentPosition();
    dx = xd.col(0) - x0;

    MatrixXd Mcurr;
    MatrixXd manips(xd.cols(), 9);
    MatrixXd errMatrix(xd.cols(),1);
    errMatrix.setZero();
    robot.startSimulation();

    vector<MatrixXd> mLoop;
    vector<double> eLoop;

    for(int i=0;i<1;i++){
//    for(int i=0;i<xd.cols();i++){
        if(i>0) dx = xd.col(i) - xd.col(i-1);
        MatrixXd MDesired = xHat.row(i);
        MDesired.resize(3,3);
//        Mcurr = ManipulabilityTrackingSecondaryTask(robot, xd.col(i), dx, MDesired);
        Mcurr=ManipulabilityTrackingMainTask(robot, MDesired, mLoop, eLoop);
        errMatrix(i,0)=(MDesired.pow(-0.5)*Mcurr*MDesired.pow(-0.5)).log().norm();
        Mcurr.resize(1,9);
        manips.row(i) = Mcurr;
        writeCSV(mLoop, "/home/nnrthmr/CLionProjects/ma_thesis/data/tracking/loopManipulabilities.csv");
    }
    deb("done");
    robot.stopSimulation();
    for(int i=0;i<manips.rows();i++) manips.row(i)=manips.row(i)/10;
    writeCSV(errMatrix, "/home/nnrthmr/CLionProjects/ma_thesis/data/tracking/human_arm/errorManipulabilities.csv");
    writeCSV(manips, "/home/nnrthmr/CLionProjects/ma_thesis/data/tracking/human_arm/xhat.csv");
}

/**
 *  Control only manipulabilities of given human arm movement
 */
void controlManipulabilitiesRHumanArm(string exp, string proband, int nPoints, int nDemos, int totalPoints){
    Franka robot = Franka(false);
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
//    MatrixXd manips(100, 9);
//    MatrixXd errMatrix(100,1);
    errMatrix.setZero();
    robot.startSimulation();

    for(int i=0;i<1;i++){
//    for(int i=0;i<100;i++){
//    for(int i=0;i<xd.cols();i++){
        if(i>0) dx = xd.col(i) - xd.col(i-1);
        MatrixXd MDesired = xHat.row(i);
        MDesired.resize(3,3);
        MDesired.setIdentity();
        MDesired=MDesired/3;
//        MDesired=MDesired/5;
//        Mcurr = ManipulabilityTrackingSecondaryTask(robot, xd.col(i), dx, MDesired);
        Mcurr=ManipulabilityTrackingMainTask(robot, MDesired, mLoop, eLoop);
        errMatrix(i,0)=(MDesired.pow(-0.5)*Mcurr*MDesired.pow(-0.5)).log().norm();
        Mcurr.resize(1,9);
        manips.row(i) = Mcurr;

        MatrixXd eLoopMat(1, eLoop.size());
        for(int i=0;i<eLoop.size();i++){
            eLoopMat(0,i)=eLoop[i];
        }
        writeCSV(mLoop, "/home/nnrthmr/CLionProjects/ma_thesis/data/tracking/rhuman/" + exp + "/" + proband +
                        "/loopManipulabilities.csv");
        writeCSV(eLoopMat, "/home/nnrthmr/CLionProjects/ma_thesis/data/tracking/rhuman/" + exp + "/" + proband +
                           "/loopErrors.csv");
    }
    deb("done");
    robot.stopSimulation();
    writeCSV(errMatrix, "/home/nnrthmr/CLionProjects/ma_thesis/data/tracking/rhuman/" + exp + "/" + proband +
                        "/errorManipulabilities.csv");
    writeCSV(manips, "/home/nnrthmr/CLionProjects/ma_thesis/data/tracking/rhuman/" + exp + "/" + proband +
                     "/controlledManipulabilities.csv");
}

int main() {
    // Load the demonstration data
//    string data_path="/home/nnrthmr/Desktop/master-thesis/vrep/vrep_franka_promps/py_scripts/data/";
//    MatrixXd data_pos(4, 2*21);
//    vector<Tensor3d> data_m;
//    load_data(data_path, 2, 21, &data_m, &data_pos);

//    std::cout.precision(20);

/**
 * Only perform the learning part and save the learned data in xHat and xd
 */
//MatrixXd xd(3, 20);
//MatrixXd xHat(6, 20);
//learn3d(xd, xHat);

/**
 * Perform learning part on Luis experiment data, RHuMAn model data
 */

//string exp="cut_userchoice";
//string proband="4";
//
//string infoPath;
//if (proband=="") infoPath = "/home/nnrthmr/Desktop/RHuMAn-arm-model/data/"+exp+"/agg/info.txt";
//else infoPath="/home/nnrthmr/Desktop/RHuMAn-arm-model/data/"+exp+"/agg/"+proband+"/info.txt";
//
//ifstream infile(infoPath);
//int nPoints, nDemos, totalPoints;
//infile >> nPoints >> nDemos >> totalPoints;
//assert(nPoints*nDemos==totalPoints);

//MatrixXd xd(3,nPoints);
//MatrixXd xHat(6,nPoints);
//learn3dRHumanMotion(xd, xHat, nPoints, nDemos, totalPoints, exp, proband);
//controlManipulabilitiesRHumanArm(exp, proband, nPoints, nDemos, totalPoints);

/**
 * Only perform the control part and use the learned data in xHat and xd
 */
//control();
//controlManipulabilitiesHumanArm();



/**
 * Transfer to another robot -> not fully implemented and tested yet
 */
//Franka robotStudent = Franka(false);
//if(dimensions==3)  transfer3d(robotStudent);

/**
 * Performing the whole pipeline from GMR to execution
 */
//learnAndConrol();

//    writeCSV(expSigma, "/home/nnrthmr/CLionProjects/ma_thesis/data/expSigma.csv");
//

/**
 * Process human arm motion example data
 * For the moment I generate 4 samples by randomly adding to given data (only every 20th entry to make smaller set).
 * First 400 entries are from real data.
 */
//MatrixXd xd(3, 8219);
//MatrixXd xHat(6, 8219);

//MatrixXd xd(3, 400);
//MatrixXd xHat(6, 400);
//learn3dHumanMotion(xd, xHat);


/**
 * Moving to a goal with a simple control loop
 */
//    Franka robot = Franka(false);
//    VectorXd q_goal(7);
//    q_goal << -pi/2.0, 0.004, 0.0, -1.57156, 0.0, 1.57075, 0.0;
//    robot.moveToQGoal(q_goal);


//MatrixXd jtfull(6,4);
//jtfull <<-2.00000000000000,	2.00000000000000,	5.46410161513775,	3.46410161513776,
//    3.46410161513776,	3.46410161513776,	1.46410161513776,	-2.00000000000000,
//    0,	0,	0,	0,
//    0,	0,	0,	0,
//    0,	0,	0,	0,
//    1,	1,	1,	1;
//
//MatrixXd med(2,2);
//med <<304.380134285343,	-188.288919971871,
//        -188.288919971871,	124.941294748085;
//
//MatrixXd mct(2,2);
//mct << 49.8564064605510,	1.07179676972449,
//        1.07179676972449,	30.1435935394490;
//
//
//vector<MatrixXd> t;
//t.push_back(med);
//MatrixXd mdiff = logMap(med, mct);
//
//MatrixXd xd(2,1);
//xd <<0,0;
//
//MatrixXd x(2,1);
//xd <<0.3,0.3;
//
//ManipulabilityTrackingSecondaryTask(robot, x, xd, med);

/**
 * Mapping functions test
 */

//MatrixXd A(3,3);
//A << 1,0,0,0,1,0,0,0,1;
//
//deb(getPrincipalAxes(A))
//deb(getLengthsOfPrincipalAxes(A))

return 0;
}
