#include <learn.h>

#define dimensions 3

void learn2d(){
    // Build GMM for trajectories in cartesian coordinates

    string cmat_path="/home/nnrthmr/CLionProjects/ma_thesis/data/C_Mat.csv";
    //MatrixXd data(3, 400);
    MatrixXd data(4, 400);
//    vector<Tensor3d> data_m;
    load_data_cmat(cmat_path, &data);
    data.row(3)=data.row(2)*0.75;

    GMM model = GMM();
    model.InitModel(data, 4);
    model.TrainEM();

    MatrixXd expData(3, 100);
    expData.setZero();
    vector<MatrixXd> expSigma; //m_dimOut x m_dimOut x m_nData
    cout<<"GMR"<<std::endl;
    model.GMR(expData, expSigma);

    writeCSV(expData, "/home/nnrthmr/CLionProjects/ma_thesis/data/expData3d.csv");
    writeCSV(expSigma, "/home/nnrthmr/CLionProjects/ma_thesis/data/expSigma3d.csv");

//     Build GMM for Manifold
    string mmat_path="/home/nnrthmr/CLionProjects/ma_thesis/data/Manip_Mat.csv";
    MatrixXd data2(400, 4);
    loadCSVSkipFirst(mmat_path, &data2);

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
    loadCSVSkipFirst(cmat_path, &data);

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
//    writeCSV(model2.m_muMan, "/home/nnrthmr/CLionProjects/ma_thesis/data/model2MuMan.csv");
//    writeCSV(model2.m_sigma, "/home/nnrthmr/CLionProjects/ma_thesis/data/model2Sigma.csv");
    writeCSV(vec2Symmat(xHat), "/home/nnrthmr/CLionProjects/ma_thesis/data/xhat.csv");
    writeCSV(xd, "/home/nnrthmr/CLionProjects/ma_thesis/data/xd.csv");
//    writeCSV(expSigma, "/home/nnrthmr/CLionProjects/ma_thesis/data/modelSigma.csv");
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
        dataVectorized.block(i, 1, 1, 6) = symmat2Vec(tmp * tmp.transpose());
        dataVectorized.block(400+i, 1, 1, 6) = symmat2Vec(tmp * tmp.transpose());
        dataVectorized.block(800+i, 1, 1, 6) = symmat2Vec(tmp * tmp.transpose());
        dataVectorized.block(1200+i, 1, 1, 6) = symmat2Vec(tmp * tmp.transpose());
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
//    writeCSV(model2.m_muMan, "/home/nnrthmr/CLionProjects/ma_thesis/data/model2MuMan.csv");
//    writeCSV(model2.m_sigma, "/home/nnrthmr/CLionProjects/ma_thesis/data/model2Sigma.csv");
    writeCSV(vec2Symmat(xHat / 10), "/home/nnrthmr/CLionProjects/ma_thesis/data/learning/human_arm/xhat.csv");
    writeCSV(xd / 10, "/home/nnrthmr/CLionProjects/ma_thesis/data/learning/human_arm/xd.csv");
    writeCSV(data / 10, "/home/nnrthmr/CLionProjects/ma_thesis/data/demos/human_arm/dummyTrajectories.csv");
    vector<MatrixXd> manips = vec2Symmat(dataVectorized.rightCols(6).transpose());
    for(int i=0;i<manips.size();i++) manips[i]=manips[i]/10;
    writeCSV(manips, "/home/nnrthmr/CLionProjects/ma_thesis/data/demos/human_arm/dummyManipulabilities.csv");
//    writeCSV(expSigma, "/home/nnrthmr/CLionProjects/ma_thesis/data/modelSigma.csv");
}

/**
 * This function learns from the experiments modeled with RHuMAn model (provided by Luis)
 * @param xd trajectory is written here
 * @param xHat manipulabilities are written here
 */
void learn3dRHumanMotion(MatrixXd &xd, MatrixXd &xHat, const int nPoints, const int nDemos, const int totalPoints, const string exp, const string proband){
    //Load data and build manipulabilities
    std::cout<<"Loading demonstrations ..."<<std::endl;

    deb(exp)
    deb(proband)

    if (mkdir(("/home/nnrthmr/CLionProjects/ma_thesis/data/learning/rhuman/"+exp).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1)
    {
        if( errno == EEXIST ) {
        } else {
            throw std::runtime_error( strerror(errno) );
        }
    }
    if (mkdir(("/home/nnrthmr/CLionProjects/ma_thesis/data/learning/rhuman/"+exp+"/"+proband).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1)
    {
        if( errno == EEXIST ) {
        } else {
            throw std::runtime_error( strerror(errno) );
        }
    }

    deb(nPoints)
    deb(nDemos)
    deb(totalPoints)

    MatrixXd data(totalPoints,4);
    MatrixXd m(totalPoints,10);
    MatrixXd dataVectorized(totalPoints, 7);

    string tPath,mPath;
    if (proband=="") tPath = "/home/nnrthmr/Desktop/RHuMAn-arm-model/data/"+exp+"/agg/all_t.csv";
    else tPath="/home/nnrthmr/Desktop/RHuMAn-arm-model/data/"+exp+"/agg/"+proband+"/all_t.csv";

    if (proband=="") mPath = "/home/nnrthmr/Desktop/RHuMAn-arm-model/data/"+exp+"/agg/all_m.csv";
    else mPath="/home/nnrthmr/Desktop/RHuMAn-arm-model/data/"+exp+"/agg/"+proband+"/all_m.csv";


//    if (proband=="") tPath = "/home/nnrthmr/CLionProjects/ma_thesis/data/demos/rhuman_luis/data/"+exp+"/interpolated/agg/all_t.csv";
//    else tPath="/home/nnrthmr/CLionProjects/ma_thesis/data/demos/rhuman_luis/data/"+exp+"/interpolated/agg/"+proband+"/all_t.csv";
//
//    if (proband=="") mPath = "/home/nnrthmr/CLionProjects/ma_thesis/data/demos/rhuman_luis/data/"+exp+"/interpolated/agg/all_m.csv";
//    else mPath="/home/nnrthmr/CLionProjects/ma_thesis/data/demos/rhuman_luis/data/"+exp+"/interpolated/agg/"+proband+"/all_m.csv";

    deb(tPath)
    deb(mPath)
    loadCSVSkipFirst(tPath, &data);
    loadCSVSkipFirst(mPath, &m);
    deb(data)
    deb(m)

    data.rightCols(3) = data.rightCols(3) * 10;
    m=m*10;

    MatrixXd tmp;
    dataVectorized.leftCols(1) = data.leftCols(1);

    for (int i = 0; i < m.rows(); i++) {
        tmp = m.row(i).rightCols(9);
        tmp.resize(3,3);
        dataVectorized.row(i).rightCols(6) = symmat2Vec(tmp);
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

    MatrixXd errors = getDiffVector(vec2Symmat(xHat), m, nPoints); //demo0
    writeCSV(errors.colwise().mean().transpose(),
             "/home/nnrthmr/CLionProjects/ma_thesis/data/learning/rhuman/" + exp + "/" + proband + "/xhatErrors.csv");
    deb(errors.mean())

    //Write model2.muMan, model2.sigma, xhat to files
//    writeCSV(model2.m_muMan, "/home/nnrthmr/CLionProjects/ma_thesis/data/model2MuMan.csv");
//    writeCSV(model2.m_sigma, "/home/nnrthmr/CLionProjects/ma_thesis/data/model2Sigma.csv");


    writeCSV(vec2Symmat(xHat / 10),
             "/home/nnrthmr/CLionProjects/ma_thesis/data/learning/rhuman/" + exp + "/" + proband + "/xhat.csv");
    writeCSV(xd / 10, "/home/nnrthmr/CLionProjects/ma_thesis/data/learning/rhuman/" + exp + "/" + proband + "/xd.csv");




//    writeCSV(data/10, "/home/nnrthmr/CLionProjects/ma_thesis/data/demos/human_arm/dummyTrajectories.csv");
//    vector<MatrixXd> manips = vec2Symmat(dataVectorized.rightCols(6).transpose());
//    writeCSV(manips, "/home/nnrthmr/CLionProjects/ma_thesis/data/demos/human_arm/dummyManipulabilities.csv");
//    writeCSV(expSigma, "/home/nnrthmr/CLionProjects/ma_thesis/data/modelSigma.csv");
}
