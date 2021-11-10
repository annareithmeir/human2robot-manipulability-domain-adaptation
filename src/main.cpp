#include <Franka.h>
#include <GMM.h>
#include <GMM_SPD.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>

using namespace std;
using namespace Eigen;

#define deb(x) cout << #x << " " << x << endl;
#define dimensions 3

void learn2d(){
    // Build GMM for trajectories in cartesian coordinates

    string cmat_path="/home/nnrthmr/CLionProjects/ma_thesis/data/C_Mat.csv";
    //MatrixXd data(3, 400);
    MatrixXd data(4, 400);
    vector<Tensor3d> data_m;
    load_data_cmat(cmat_path, &data);
    data.row(3)=data.row(2)*0.75;

    GMM model = GMM();
    model.InitModel(data, 4);
    model.TrainEM();

    MatrixXd expData(3, 100);
    expData.setZero();
    std::vector<MatrixXd> expSigma; //m_dimOut x m_dimOut x m_nData
    std::cout<<"GMR"<<std::endl;
    model.GMR(expData, expSigma);

    WriteCSV(expData, "/home/nnrthmr/CLionProjects/ma_thesis/data/expData3d.csv");
    WriteCSV(expSigma, "/home/nnrthmr/CLionProjects/ma_thesis/data/expSigma3d.csv");

//     Build GMM for Manifold
    string mmat_path="/home/nnrthmr/CLionProjects/ma_thesis/data/Manip_Mat.csv";
    MatrixXd data2(400, 4);
    load_data_mmat_skip_first(mmat_path, &data2);

    GMM_SPD model2 = GMM_SPD();
    model2.InitModel(data2, 4);
    model2.TrainEM();

    MatrixXd expData2(3, 100);
    expData.setZero();
    std::vector<MatrixXd> expSigma2; //m_dimOut x m_dimOut x m_nData

    std::cout<<"GMR"<<std::endl;
    model2.GMR(expData2, expSigma2);
}

/**
 * This function learns from the simulated 3d data from VREP
 * @param xd trajectory is written here
 * @param xHat manipulabilities are written here
 */
void learn3d(MatrixXd &xd, MatrixXd &xHat){
    // Build GMM for trajectories in cartesian coordinates

    std::cout<<"Loading demonstrations ..."<<std::endl;
    string cmat_path="/home/nnrthmr/CLionProjects/ma_thesis/data/demos/trajectories.csv";
    MatrixXd data(80, 4);
    load_data_mmat_skip_first(cmat_path, &data);

    string mmat_path = "/home/nnrthmr/CLionProjects/ma_thesis/data/demos/translationManip3d.csv";
    MatrixXd data2(80, 10);
    MatrixXd dataVectorized(80, 7);
    load_data_mmat_skip_first(mmat_path, &data2);

    for (int i = 0; i < data2.rows(); i++) {
        dataVectorized(i, 0) = data2(i,0);
        MatrixXd tmp = data2.block(i, 1, 1, 9);
        tmp.resize(3, 3);
        dataVectorized.block(i, 1, 1, 6) = Symmat2Vec(tmp);
    }

    std::cout<<"GMM  for trajectories ..."<<std::endl;
    GMM model = GMM();
    model.InitModel(data.transpose(), 4);
    model.TrainEM();

//    MatrixXd xd(3, 20);
    xd.setZero();
    std::vector<MatrixXd> expSigma; //m_dimOut x m_dimOut x m_nData
    model.GMR(xd, expSigma);

    std::cout<<"GMM for manipulabilities ..."<<std::endl;
    GMM_SPD model2 = GMM_SPD();
    model2.InitModel(dataVectorized, 4);
    model2.TrainEM();

//    MatrixXd xHat(6, 20);
    xHat.setZero();
    std::vector<MatrixXd> expSigma2; //m_dimOut x m_dimOut x m_nData

    model2.GMR(xHat, expSigma2);

    //Write model2.muMan, model2.sigma, xhat to files
//    WriteCSV(model2.m_muMan, "/home/nnrthmr/CLionProjects/ma_thesis/data/model2MuMan.csv");
//    WriteCSV(model2.m_sigma, "/home/nnrthmr/CLionProjects/ma_thesis/data/model2Sigma.csv");
    WriteCSV(Vec2Symmat(xHat), "/home/nnrthmr/CLionProjects/ma_thesis/data/xhat.csv");
    WriteCSV(xd, "/home/nnrthmr/CLionProjects/ma_thesis/data/xd.csv");
//    WriteCSV(expSigma, "/home/nnrthmr/CLionProjects/ma_thesis/data/modelSigma.csv");
}

/**
 * This function learns from the human motion data from Luis
 * @param xd trajectory is written here
 * @param xHat manipulabilities are written here
 */
void learn3dHumanMotion(MatrixXd &xd, MatrixXd &xHat){
    //Load data and build manipulabilities
    std::cout<<"Loading demonstrations ..."<<std::endl;
    MatrixXd pos(8219,3);
    MatrixXd posJac(8219,36);
    readTxtFile("/home/nnrthmr/CLionProjects/ma_thesis/data/demos/human_arm/humanArmMotionOutput.txt", &pos, &posJac);

    // Generate dummy data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-0.00001, 0.00001);

    MatrixXd randPos = Eigen::MatrixXf::Zero(1200,3).unaryExpr([&](double dummy){return dis(gen);});
    MatrixXd randM = Eigen::MatrixXf::Zero(1200,6).unaryExpr([&](double dummy){return dis(gen);});


//    MatrixXd data(8219, 4);
//    MatrixXd dataVectorized(8219, 7);
//    MatrixXd data(400, 4);
//    MatrixXd dataVectorized(400, 7);
    MatrixXd data(1600, 4);
    MatrixXd dataVectorized(1600, 7);
    int tmpEveryNth = 20;

    MatrixXd tmp;
    for (int i = 0; i < 400; i++) {
        dataVectorized(i, 0) = i*0.01+0.01;
        dataVectorized(400+i, 0) = i*0.01+0.01;
        dataVectorized(800+i, 0) = i*0.01+0.01;
        dataVectorized(1200+i, 0) = i*0.01+0.01;
        data(i, 0) = i*0.01+0.01;
        data(400+i, 0) = i*0.01+0.01;
        data(800+i, 0) = i*0.01+0.01;
        data(1200+i, 0) = i*0.01+0.01;
//        data.block(i,1,1,3) = pos.row(i*20); // skim dataset to every 20th element
//        data.block(400+i,1,1,3) = pos.row(i*20) + randPos.row(i);
//        data.block(800+i,1,1,3) = pos.row(i*20) + randPos.row(400+i);
//        data.block(1200+i,1,1,3) = pos.row(i*20) + randPos.row(800+i);
//        tmp = posJac.row(i*20);
        data.block(i,1,1,3) = pos.row(i*tmpEveryNth); // skim dataset to first 400 elements
        data.block(400+i,1,1,3) = pos.row(i*tmpEveryNth) + randPos.row(i);
        data.block(800+i,1,1,3) = pos.row(i*tmpEveryNth) + randPos.row(400+i);
        data.block(1200+i,1,1,3) = pos.row(i*tmpEveryNth) + randPos.row(800+i);
        tmp = posJac.row(i*tmpEveryNth)*10;
        tmp.resize(3, 12);
        dataVectorized.block(i, 1, 1, 6) = Symmat2Vec(tmp*tmp.transpose());
        dataVectorized.block(400+i, 1, 1, 6) = Symmat2Vec(tmp*tmp.transpose());
        dataVectorized.block(800+i, 1, 1, 6) = Symmat2Vec(tmp*tmp.transpose());
        dataVectorized.block(1200+i, 1, 1, 6) = Symmat2Vec(tmp*tmp.transpose());
    }
    data.rightCols(3) = data.rightCols(3) * 10;
    deb(data.transpose());

    // Build GMM for trajectories in cartesian coordinates
    std::cout<<"GMM  for trajectories ..."<<std::endl;
    GMM model = GMM();
    model.InitModel(data.transpose(), 4);
    model.TrainEM();

//    MatrixXd xd(3, 20);
    xd.setZero();
    std::vector<MatrixXd> expSigma; //m_dimOut x m_dimOut x m_nData
    model.GMR(xd, expSigma);

    std::cout<<"GMM for manipulabilities ..."<<std::endl;
    GMM_SPD model2 = GMM_SPD();
    model2.InitModel(dataVectorized, 4);
    model2.TrainEM();

//    MatrixXd xHat(6, 20);
    xHat.setZero();
    std::vector<MatrixXd> expSigma2; //m_dimOut x m_dimOut x m_nData

    model2.GMR(xHat, expSigma2);
    deb(data.transpose());
    deb(xd);
    deb(dataVectorized.transpose());
    deb(xHat);


    //Write model2.muMan, model2.sigma, xhat to files
//    WriteCSV(model2.m_muMan, "/home/nnrthmr/CLionProjects/ma_thesis/data/model2MuMan.csv");
//    WriteCSV(model2.m_sigma, "/home/nnrthmr/CLionProjects/ma_thesis/data/model2Sigma.csv");
    WriteCSV(Vec2Symmat(xHat/10), "/home/nnrthmr/CLionProjects/ma_thesis/data/results/human_arm/xhat.csv");
    WriteCSV(xd/10, "/home/nnrthmr/CLionProjects/ma_thesis/data/results/human_arm/xd.csv");
    WriteCSV(data/10, "/home/nnrthmr/CLionProjects/ma_thesis/data/demos/human_arm/dummyTrajectories.csv");
    vector<MatrixXd> manips = Vec2Symmat(dataVectorized.rightCols(6).transpose());
    for(int i=0;i<manips.size();i++) manips[i]=manips[i]/10;
    WriteCSV(manips, "/home/nnrthmr/CLionProjects/ma_thesis/data/demos/human_arm/dummyManipulabilities.csv");
//    WriteCSV(expSigma, "/home/nnrthmr/CLionProjects/ma_thesis/data/modelSigma.csv");
}

/**
 * This function learns from the experiments modeled with RHuMAn model (provided by Luis)
 * @param xd trajectory is written here
 * @param xHat manipulabilities are written here
 */
void learn3dRHumanMotion(MatrixXd &xd, MatrixXd &xHat, int nPoints, int nDemos, int totalPoints, string exp, string proband){
    //Load data and build manipulabilities
    std::cout<<"Loading demonstrations ..."<<std::endl;

    deb(exp)
    deb(proband)

    if (mkdir(("/home/nnrthmr/CLionProjects/ma_thesis/data/results/rhuman/"+exp).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1)
    {
        if( errno == EEXIST ) {
        } else {
            throw std::runtime_error( strerror(errno) );
        }
    }
    if (mkdir(("/home/nnrthmr/CLionProjects/ma_thesis/data/results/rhuman/"+exp+"/"+proband).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1)
    {
        if( errno == EEXIST ) {
        } else {
            throw std::runtime_error( strerror(errno) );
        }
    }

    deb(nPoints)
    deb(nDemos)
    deb(totalPoints)

    MatrixXd data(totalPoints,4); //25 * 1200 drill_optimal
    MatrixXd m(totalPoints,10);
    MatrixXd dataVectorized(totalPoints, 7);

    string tPath,mPath;
    if (proband=="") tPath = "/home/nnrthmr/Desktop/RHuMAn-arm-model/data/"+exp+"/agg/all_t.csv";
    else tPath="/home/nnrthmr/Desktop/RHuMAn-arm-model/data/"+exp+"/agg/"+proband+"/all_t.csv";

    if (proband=="") mPath = "/home/nnrthmr/Desktop/RHuMAn-arm-model/data/"+exp+"/agg/all_m.csv";
    else mPath="/home/nnrthmr/Desktop/RHuMAn-arm-model/data/"+exp+"/agg/"+proband+"/all_m.csv";

    deb(tPath)
    deb(mPath)
    load_data_mmat_skip_first(tPath, &data);
    load_data_mmat_skip_first(mPath, &m);
    deb(data.topRows(3))
    deb(m.topRows(3))

    MatrixXd tmp;
    dataVectorized.leftCols(1) = data.leftCols(1);

    for (int i = 0; i < m.rows(); i++) {
        MatrixXd tmp = m.row(i).rightCols(9);
        tmp.resize(3,3);
        dataVectorized.row(i).rightCols(6) = Symmat2Vec(tmp);
    }

    // Build GMM for trajectories in cartesian coordinates
    std::cout<<"GMM  for trajectories ..."<<std::endl;
    GMM model = GMM();
    model.InitModel(data.transpose(), nDemos);
    model.TrainEM();

    xd.setZero();
    std::vector<MatrixXd> expSigma; //m_dimOut x m_dimOut x m_nData
    model.GMR(xd, expSigma);

    std::cout<<"GMM for manipulabilities ..."<<std::endl;
    GMM_SPD model2 = GMM_SPD();
    model2.InitModel(dataVectorized, nDemos);
    model2.TrainEM();

    xHat.setZero();
    std::vector<MatrixXd> expSigma2; //m_dimOut x m_dimOut x m_nData

    model2.GMR(xHat, expSigma2);
    deb(data.transpose());
    deb(xd);
    deb(dataVectorized.transpose());
    deb(xHat);

    deb(model2.m_muMan)



    //Write model2.muMan, model2.sigma, xhat to files
//    WriteCSV(model2.m_muMan, "/home/nnrthmr/CLionProjects/ma_thesis/data/model2MuMan.csv");
//    WriteCSV(model2.m_sigma, "/home/nnrthmr/CLionProjects/ma_thesis/data/model2Sigma.csv");
    WriteCSV(Vec2Symmat(xHat), "/home/nnrthmr/CLionProjects/ma_thesis/data/results/rhuman/"+exp+"/"+proband+"/xhat.csv");
    WriteCSV(xd, "/home/nnrthmr/CLionProjects/ma_thesis/data/results/rhuman/"+exp+"/"+proband+"/xd.csv");
//    WriteCSV(data/10, "/home/nnrthmr/CLionProjects/ma_thesis/data/demos/human_arm/dummyTrajectories.csv");
//    vector<MatrixXd> manips = Vec2Symmat(dataVectorized.rightCols(6).transpose());
//    WriteCSV(manips, "/home/nnrthmr/CLionProjects/ma_thesis/data/demos/human_arm/dummyManipulabilities.csv");
//    WriteCSV(expSigma, "/home/nnrthmr/CLionProjects/ma_thesis/data/modelSigma.csv");
}

void transfer3d(Franka robotStudent){
    // Build GMM for trajectories in cartesian coordinates
    string cmat_path="/home/nnrthmr/CLionProjects/ma_thesis/data/demos/trajectories.csv";
    MatrixXd data(80, 4);
    load_data_mmat_skip_first(cmat_path, &data);

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
    load_data_mmat_skip_first(mmat_path, &data2);

    for (int i = 0; i < data2.rows(); i++) {
        dataVectorized(i, 0) = data2(i,0);
        MatrixXd tmp = data2.block(i, 1, 1, 9);
        tmp.resize(3, 3);
        dataVectorized.block(i, 1, 1, 6) = Symmat2Vec(tmp);
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
        mDiff = LogMap(Vec2Symmat(xHat.col(t))[0], ManipCurr); //only one element because colwise

        //manipulability tracking in nullspace of position command
        pinv=Jmt.completeOrthogonalDecomposition().pseudoInverse();
        mCommandNs = pinv * Symmat2Vec(mDiff).transpose();
        pinv=Jt.completeOrthogonalDecomposition().pseudoInverse();
        dqns=(MatrixXd::Identity(robotStudent.m_dof,robotStudent.m_dof) - pinv*Jt) * model2.m_km * mCommandNs;
        qt = qt + ( dqT1 + dqns )* model2.m_dt;
        deb(qt);
//        robotStudent.setJoints(qt);
    }
    std::cout<<"GMR"<<std::endl;

    //Write model2.muMan, model2.sigma, xhat to files
    WriteCSV(model2.m_muMan, "/home/nnrthmr/CLionProjects/ma_thesis/data/model2MuMan.csv");
    WriteCSV(model2.m_sigma, "/home/nnrthmr/CLionProjects/ma_thesis/data/model2Sigma.csv");
    WriteCSV(Vec2Symmat(xHat), "/home/nnrthmr/CLionProjects/ma_thesis/data/xhat.csv");
}

/**
 * This function learns from the simulated 3d data from VREP and then controls the robot in VREP to follow
 * according to the learned data.
 */
void learnAndConrol(){
    Franka robot = Franka();
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
        Mcurr = robot.ManipulabilityTrackingSecondaryTask(xd.col(i), dx, Vec2Symmat(xHat.col(i))[0]);

//        Mcurr=robot.ManipulabilityTrackingMainTask(Vec2Symmat(xHat.col(i))[0]);
//        saveData.block(i,0,1,3) = Vec2Symmat(xHat.col(i))[0].row(0);
//        saveData.block(i,3,1,3) = Vec2Symmat(xHat.col(i))[0].row(1);
//        saveData.block(i,6,1,3) = Vec2Symmat(xHat.col(i))[0].row(2);
//        saveData.block(i,9,1,3) = Mcurr.row(0);
//        saveData.block(i,12,1,3) = Mcurr.row(1);
//        saveData.block(i,15,1,3) = Mcurr.row(2);
    }
    deb("done");

//    WriteCSV(saveData, "/home/nnrthmr/CLionProjects/ma_thesis/data/tracking/test.csv");
}


/**
 * This function learns from the simulated 3d data from VREP and then controls the robot in VREP to follow
 * according to the learned data.
 */
void control(){
    Franka robot = Franka();
    MatrixXd xdTmp(80,4);
    MatrixXd xd(3, 20);
    MatrixXd xHat(20,9);
    load_data_mmat_skip_first("/home/nnrthmr/CLionProjects/ma_thesis/data/demos/trajectories.csv", &xdTmp);
    xd=xdTmp.rightCols(3).topRows(20).transpose();
    load_data_mmat("/home/nnrthmr/CLionProjects/ma_thesis/data/xhat.csv", &xHat);

    VectorXd dx(3);
    VectorXd x0(3);
    x0 = robot.getCurrentPosition();
    dx = xd.col(0) - x0;

    vector<MatrixXd> mLoop;
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
//        Mcurr = robot.ManipulabilityTrackingSecondaryTask(xd.col(i), dx, MDesired);
        Mcurr=robot.ManipulabilityTrackingMainTask(MDesired, mLoop);
        errMatrix(i,0)=(MDesired.pow(-0.5)*Mcurr*MDesired.pow(-0.5)).log().norm();
        Mcurr.resize(1,9);
        manips.row(i) = Mcurr;
    }
    deb("done");
    robot.stopSimulation();

    WriteCSV(errMatrix, "/home/nnrthmr/CLionProjects/ma_thesis/data/tracking/errorManipulabilities.csv");
    WriteCSV(manips, "/home/nnrthmr/CLionProjects/ma_thesis/data/tracking/xhat.csv");
}
/**
 *  Control only manipulabilities of given human arm movement
 */
void controlManipulabilitiesHumanArm(){
    Franka robot = Franka();
    MatrixXd xdTmp(1600,4);
    MatrixXd xhatTmp(1600,9);
    MatrixXd xd(3, 400);
    MatrixXd xHat(400,9);
    load_data_mmat("/home/nnrthmr/CLionProjects/ma_thesis/data/demos/human_arm/dummyTrajectories.csv", &xdTmp);
    xd=xdTmp.rightCols(3).topRows(400).transpose();
    xd=xd*10;
    load_data_mmat("/home/nnrthmr/CLionProjects/ma_thesis/data/demos/human_arm/dummyManipulabilities.csv", &xhatTmp);
    xHat = xhatTmp.topRows(400); // TODO try scaling by 10, upper bound for gain
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

//    for(int i=0;i<1;i++){
    for(int i=0;i<xd.cols();i++){
        if(i>0) dx = xd.col(i) - xd.col(i-1);
        MatrixXd MDesired = xHat.row(i);
        MDesired.resize(3,3);
//        Mcurr = robot.ManipulabilityTrackingSecondaryTask(xd.col(i), dx, MDesired);
        Mcurr=robot.ManipulabilityTrackingMainTask(MDesired, mLoop);
        errMatrix(i,0)=(MDesired.pow(-0.5)*Mcurr*MDesired.pow(-0.5)).log().norm();
        Mcurr.resize(1,9);
        manips.row(i) = Mcurr;
        WriteCSV(mLoop, "/home/nnrthmr/CLionProjects/ma_thesis/data/tracking/loopManipulabilities.csv");
    }
    deb("done");
    robot.stopSimulation();
    for(int i=0;i<manips.rows();i++) manips.row(i)=manips.row(i)/10;
    WriteCSV(errMatrix, "/home/nnrthmr/CLionProjects/ma_thesis/data/tracking/human_arm/errorManipulabilities.csv");
    WriteCSV(manips, "/home/nnrthmr/CLionProjects/ma_thesis/data/tracking/human_arm/xhat.csv");
}

/**
 *  Control only manipulabilities of given human arm movement
 */
void controlManipulabilitiesRHumanArm(string exp, string proband, int nPoints, int nDemos, int totalPoints){
    Franka robot = Franka();
    MatrixXd xdTmp(nPoints,4);
    MatrixXd xhatTmp(nPoints,9);
    MatrixXd xd(3, nPoints);
    MatrixXd xHat(nPoints,9);
    load_data_mmat("/home/nnrthmr/CLionProjects/ma_thesis/data/results/rhuman/"+exp+"/"+proband+"/xd.csv", &xdTmp);
    xd=xdTmp.rightCols(3).transpose();
    load_data_mmat("/home/nnrthmr/CLionProjects/ma_thesis/data/results/rhuman/"+exp+"/"+proband+"/xhat.csv", &xhatTmp);
    xHat = xhatTmp;

    vector<MatrixXd> mLoop;

    VectorXd dx(3);
    VectorXd x0(3);
    x0 = robot.getCurrentPosition();
    dx = xd.col(0) - x0;

    MatrixXd Mcurr;
    MatrixXd manips(xd.cols(), 9);
    MatrixXd errMatrix(xd.cols(),1);
    errMatrix.setZero();
    robot.startSimulation();

    for(int i=0;i<1;i++){
//    for(int i=0;i<xd.cols();i++){
        if(i>0) dx = xd.col(i) - xd.col(i-1);
        MatrixXd MDesired = xHat.row(i);
        MDesired.resize(3,3);
//        Mcurr = robot.ManipulabilityTrackingSecondaryTask(xd.col(i), dx, MDesired);
        Mcurr=robot.ManipulabilityTrackingMainTask(MDesired, mLoop);
        errMatrix(i,0)=(MDesired.pow(-0.5)*Mcurr*MDesired.pow(-0.5)).log().norm();
        Mcurr.resize(1,9);
        manips.row(i) = Mcurr;
        WriteCSV(mLoop, "/home/nnrthmr/CLionProjects/ma_thesis/data/tracking/rhuman/"+exp+"/"+proband+"/loopManipulabilities.csv");
    }
    deb("done");
    robot.stopSimulation();
    WriteCSV(errMatrix, "/home/nnrthmr/CLionProjects/ma_thesis/data/tracking/rhuman/"+exp+"/"+proband+"/errorManipulabilities.csv");
    WriteCSV(manips, "/home/nnrthmr/CLionProjects/ma_thesis/data/tracking/rhuman/"+exp+"/"+proband+"/xhat.csv");
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

string exp="cut_userchoice";
string proband="5";

string infoPath;
if (proband=="") infoPath = "/home/nnrthmr/Desktop/RHuMAn-arm-model/data/"+exp+"/agg/info.txt";
else infoPath="/home/nnrthmr/Desktop/RHuMAn-arm-model/data/"+exp+"/agg/"+proband+"/info.txt";

ifstream infile(infoPath);
int nPoints, nDemos, totalPoints;
infile >> nPoints >> nDemos >> totalPoints;
assert(nPoints*nDemos==totalPoints);

MatrixXd xd(3,nPoints);
MatrixXd xHat(6,nPoints);
learn3dRHumanMotion(xd, xHat, nPoints, nDemos, totalPoints, exp, proband);

/**
 * Only perform the control part and use the learned data in xHat and xd
 */
//control();
//controlManipulabilitiesHumanArm();


/**
 * Transfer to another robot -> not fully implemented and tested yet
 */
//Franka robotStudent = Franka();
//if(dimensions==3)  transfer3d(robotStudent);

/**
 * Performing the whole pipeline from GMR to execution
 */
//learnAndConrol();

//    WriteCSV(expSigma, "/home/nnrthmr/CLionProjects/ma_thesis/data/expSigma.csv");
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
//    Franka robot = Franka();
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
//MatrixXd mdiff = LogMap(med, mct);
//
//MatrixXd xd(2,1);
//xd <<0,0;
//
//MatrixXd x(2,1);
//xd <<0.3,0.3;
//
//robot.ManipulabilityTrackingSecondaryTask(x, xd, med);

return 0;
}
