#include <Franka.h>
#include <GMM.h>
#include <GMM_SPD.h>
#include <control.h>
#include <learn.h>
#include <errno.h>
#include <sys/stat.h>

using namespace std;
using namespace Eigen;

#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60
const string robots[4]={"human", "panda", "fanuc", "puma560"};

void printProgress(double percentage) {
    int val = (int) (percentage * 100);
    int lpad = (int) (percentage * PBWIDTH);
    int rpad = PBWIDTH - lpad;
    printf("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
    fflush(stdout);
}


void generate_human_robot_data_random(string base_path, int num, double shoulder_height, int robot_type){

//    string base_path = "/home/nnrthmr/PycharmProjects/ma_thesis/5000";

    if (mkdir((base_path+"/data").c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1)
    {
        if( errno == EEXIST ) {
        } else {
            throw std::runtime_error( strerror(errno) );
        }
    }

    if (mkdir((base_path+"/data/human").c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1)
    {
        if( errno == EEXIST ) {
        } else {
            throw std::runtime_error( strerror(errno) );
        }

    }
    else{
        cout<< " Generating human data ..."<<endl;

        unique_ptr<MATLABEngine> matlabPtr = startMATLAB();
        matlab::data::ArrayFactory factory;
        matlab::data::TypedArray<float>  args_shoulder_height = factory.createScalar<float>(shoulder_height);
        matlab::data::TypedArray<int>  args_num = factory.createScalar<int>(num);
        matlab::data::CharArray args_base_path_h = factory.createCharArray(base_path+"/data/human");
        matlabPtr->setVariable(u"shoulder_height_m", std::move(args_shoulder_height));
        matlabPtr->setVariable(u"num_m", std::move(args_num));
        matlabPtr->setVariable(u"base_path_h_m", std::move(args_base_path_h));
        matlabPtr->eval(u"generateHumanData(shoulder_height_m, num_m, base_path_h_m);");
    }


    if (mkdir((base_path+"/results").c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1)
    {
        if( errno == EEXIST ) {
        } else {
            throw std::runtime_error( strerror(errno) );
        }
    }


    if(robot_type==1){ // franka panda
        if (mkdir((base_path+"/data/panda").c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1)
        {
            if( errno == EEXIST ) {
            } else {
                throw std::runtime_error( strerror(errno) );
            }
        }


        if (mkdir((base_path+"/results/panda").c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1)
        {
            if( errno == EEXIST ) {
            } else {
                throw std::runtime_error( strerror(errno) );
            }
        }

        string manips_normalized_path=base_path+"/data/panda/r_manipulabilities_normalized.csv";
        string manips_path=base_path+"/data/panda/r_manipulabilities.csv";
        string scales_path=base_path+"/data/panda/r_scales.csv";
        string positions_path=base_path+"/data/panda/r_positions.csv";
        string scales_normalized_path=base_path+"/data/panda/r_scales_normalized.csv";

        Franka robot = Franka(false);
        MatrixXd positions(num, 3);
        MatrixXd scales(num,1);
        MatrixXd scalesNormalized(num,1);
        MatrixXd manipsNormalized(num,9);
        MatrixXd manips(num,9);

        MatrixXd JFull, Jgeo, M, Mnormalized, Mresized;
        MatrixXd randomJoints = robot.GetRandomJointConfig(num); // num x 7
        VectorXd jointsCurr(7);

        cout<< " Generating robot data (Franka Emika Panda)..."<<endl;

        for(int i=0;i<num;++i){
            printProgress((float) (i)/ (float) num);
            jointsCurr= randomJoints.row(i).transpose();

            // Positions
            positions.row(i) = robot.getCurrentPosition(jointsCurr);

            // Compute manipulabilities
            JFull = robot.getPoseJacobian(jointsCurr);
            Jgeo = robot.buildGeometricJacobian(JFull, jointsCurr);
            M=Jgeo.bottomRows(3)*Jgeo.bottomRows(3).transpose();
            Mresized=M;
            Mresized.resize(1,9);
            manips.row(i) = Mresized;

            // Normalize manipulabilities to volume = 1
            double vol= getEllipsoidVolume(M);
            Mnormalized = scaleEllipsoidVolume(M, 1/vol);
            assert(getEllipsoidVolume(Mnormalized)-1<1e-4);
            Mnormalized.resize(1,9);
            manipsNormalized.row(i) = Mnormalized;
            scales(i,0) = vol;
        }

        // Normalize scales
        scalesNormalized = (scales.array()-scales.minCoeff())/(scales.maxCoeff()-scales.minCoeff());
        assert(scalesNormalized.minCoeff()>=0 && scalesNormalized.maxCoeff()<=1);

        writeCSV(manipsNormalized, manips_normalized_path);
        writeCSV(manips, manips_path);
        writeCSV(scales, scales_path);
        writeCSV(positions, positions_path);
        writeCSV(scalesNormalized, scales_normalized_path);

        // Create Lookup table
        unique_ptr<MATLABEngine> matlabPtr = startMATLAB();
        matlab::data::ArrayFactory factory;
        matlab::data::CharArray args_base_path = factory.createCharArray(base_path);
        matlab::data::CharArray args_base_path_r = factory.createCharArray(base_path+"/data/panda");
        matlab::data::CharArray args_base_path_h = factory.createCharArray(base_path+"/data/human");
        matlabPtr->setVariable(u"base_path_r_m", std::move(args_base_path_r));
        matlabPtr->setVariable(u"base_path_h_m", std::move(args_base_path_h));

        matlabPtr->eval(u"createLookupTable(base_path_h_m,base_path_r_m);");
    }
    else if(robot_type==2){ // fanuc
        if (mkdir((base_path+"/data/fanuc").c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1)
        {
            if( errno == EEXIST ) {
            } else {
                throw std::runtime_error( strerror(errno) );
            }
        }

        if (mkdir((base_path+"/results/fanuc").c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1)
        {
            if( errno == EEXIST ) {
            } else {
                throw std::runtime_error( strerror(errno) );
            }
        }
        cout<< " Generating robot data (Fanuc) ..."<<endl;

        unique_ptr<MATLABEngine> matlabPtr = startMATLAB();
        matlab::data::ArrayFactory factory;
        matlab::data::TypedArray<int>  args_num = factory.createScalar<int>(num);
        matlab::data::CharArray args_base_path = factory.createCharArray(base_path);
        matlab::data::CharArray args_base_path_r = factory.createCharArray(base_path+"/data/fanuc");
        matlab::data::CharArray args_base_path_h = factory.createCharArray(base_path+"/data/human");
        matlab::data::CharArray args_base_path_results = factory.createCharArray(base_path+"/results/fanuc");
        matlabPtr->setVariable(u"num_m", std::move(args_num));
        matlabPtr->setVariable(u"base_path_r_m", std::move(args_base_path_r));
        matlabPtr->setVariable(u"base_path_m", std::move(args_base_path));
        matlabPtr->setVariable(u"base_path_h_m", std::move(args_base_path_h));
        matlabPtr->setVariable(u"base_path_results_m", std::move(args_base_path_results));
        matlabPtr->eval(u"generateRobotDataFanuc(num_m, base_path_r_m);");
        matlabPtr->eval(u"createLookupTable(base_path_h_m,base_path_r_m);");
    }
    else if(robot_type==3){ // puma560
        if (mkdir((base_path+"/data/puma560").c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1)
        {
            if( errno == EEXIST ) {
            } else {
                throw std::runtime_error( strerror(errno) );
            }
        }

        if (mkdir((base_path+"/results/puma560").c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1)
        {
            if( errno == EEXIST ) {
            } else {
                throw std::runtime_error( strerror(errno) );
            }
        }
        cout<< " Generating robot data (PUMA560) ..."<<endl;

        unique_ptr<MATLABEngine> matlabPtr = startMATLAB();
        matlab::data::ArrayFactory factory;
        matlab::data::TypedArray<int>  args_num = factory.createScalar<int>(num);
        matlab::data::CharArray args_base_path = factory.createCharArray(base_path);
        matlab::data::CharArray args_base_path_r = factory.createCharArray(base_path+"/data/puma560");
        matlab::data::CharArray args_base_path_h = factory.createCharArray(base_path+"/data/human");
        matlab::data::CharArray args_base_path_results = factory.createCharArray(base_path+"/results/puma560");
        matlabPtr->setVariable(u"num_m", std::move(args_num));
        matlabPtr->setVariable(u"base_path_r_m", std::move(args_base_path_r));
        matlabPtr->setVariable(u"base_path_m", std::move(args_base_path));
        matlabPtr->setVariable(u"base_path_h_m", std::move(args_base_path_h));
        matlabPtr->setVariable(u"base_path_results_m", std::move(args_base_path_results));
        matlabPtr->eval(u"generateRobotDataPUMA560(num_m, base_path_r_m);");
        matlabPtr->eval(u"createLookupTable(base_path_h_m,base_path_r_m);");
    }
    else{
        cout << "No correct robot type specified. Must be 0 (human), 1 (panda) 2 (fanuc) or 3 (puma560)!"<<endl;
    }

}

void create_lookup_table_from_source_to_target(string base_path, int source, int target){
    assert(source!=target && source >=0 && target >= 0 && source < 4 && target < 4);
    cout<< " Generating lookup table from " << robots[source] << " to " << robots[target]<<" ..."<<endl;

    unique_ptr<MATLABEngine> matlabPtr = startMATLAB();
    matlab::data::ArrayFactory factory;
    matlab::data::CharArray args_base_path_r = factory.createCharArray(base_path+"/data/"+robots[target]);
    matlab::data::CharArray args_base_path_h = factory.createCharArray(base_path+"/data/"+robots[source]);
    matlabPtr->setVariable(u"base_path_r_m", std::move(args_base_path_r));
    matlabPtr->setVariable(u"base_path_h_m", std::move(args_base_path_h));
    matlabPtr->eval(u"createLookupTable(base_path_h_m,base_path_r_m);");
}

void mapManipulabilitiesNaive(string base_path){
//    string base_path = "/home/nnrthmr/PycharmProjects/ma_thesis/5000";
    unique_ptr<MATLABEngine> matlabPtr = startMATLAB();
    matlab::data::ArrayFactory factory;
    matlab::data::CharArray args_base_path = factory.createCharArray(base_path);
    matlabPtr->setVariable(u"base_path_m", std::move(args_base_path));
    matlabPtr->eval(u"map_manipulabilities(base_path_m);");
}

void mapManipulabilitiesICP(string base_path_py){
    string syscall = "cd /home/nnrthmr/PycharmProjects/ma_thesis/venv3-6/ && . bin/activate && python /home/nnrthmr/PycharmProjects/ma_thesis/run_rpa.py "+base_path_py+" && deactivate";
    system(syscall.c_str());
}

void plot(string base_path_py){
//    string syscall1 = "cd /home/nnrthmr/PycharmProjects/ma_thesis/venv3-6/ && . bin/activate && python /home/nnrthmr/PycharmProjects/ma_thesis/plot_2d_embeddings.py "+base_path_py+" && deactivate";
//    system(syscall1.c_str());
    string syscall2 = "cd /home/nnrthmr/PycharmProjects/ma_thesis/venv3-6/ && . bin/activate && python /home/nnrthmr/PycharmProjects/ma_thesis/plot_3d_ellipsoids.py "+base_path_py+" && deactivate";
    system(syscall2.c_str());
}

int main(){
//    string base_path = "/home/nnrthmr/test";
    string base_path = "/home/nnrthmr/PycharmProjects/ma_thesis/5000";
    int robot_type = 0;
    if(!fileExists(base_path+"/data/human/h_manipulabilities.csv")
        || !fileExists(base_path+"/data/"+robots[robot_type]+"/r_manipulabilities.csv"))
        generate_human_robot_data_random(base_path, 5000, 1.35, 0);
    else
        cout<< "Data already generated ..." << endl;

//    create_lookup_table_from_source_to_target(base_path, 1, 2); // lookup between r1 and r2

    mapManipulabilitiesNaive(base_path);
//    mapManipulabilitiesICP("/home/nnrthmr/PycharmProjects/ma_thesis/5000");
//    plot("/home/nnrthmr/PycharmProjects/ma_thesis/5000");


//        string manips_normalized_path = "/home/nnrthmr/PycharmProjects/ma_thesis/data/r_manipulabilities_normalized.csv";
//        string manips_path = "/home/nnrthmr/PycharmProjects/ma_thesis/data/r_manipulabilities.csv";
//        string scales_path = "/home/nnrthmr/PycharmProjects/ma_thesis/data/r_scales.csv";
//        string scales_normalized_path = "/home/nnrthmr/PycharmProjects/ma_thesis/data/r_scales_normalized.csv";
//
////    string manips_normalized_path="/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/affineTrafo/r_manipulabilities_normalized.csv";
////    string positions_path="/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/affineTrafo/r_positions.csv";
////    string scales_normalized_path="/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/affineTrafo/r_scales_normalized.csv";
//
//        string robot_manips_path = "/home/nnrthmr/PycharmProjects/ma_thesis/data/r_manipulabilities.csv";
//        MatrixXd manips(500, 9);
//        loadCSV(robot_manips_path, &manips);
//        int num = 500;
//        Franka robot = Franka(false);
//        MatrixXd positions(num, 3);
//        MatrixXd scales(num, 1);
//        MatrixXd scalesNormalized(num, 1);
//        MatrixXd manipsNormalized(num, 9);
////    MatrixXd manips(num,9);
//
//        MatrixXd JFull, Jgeo, M, Mnormalized, Mresized;
////    MatrixXd randomJoints = robot.GetRandomJointConfig(num); // num x 7
//        VectorXd jointsCurr(7);
//
//        for (int i = 0; i < num; ++i) {
//            deb(i)
//
////        jointsCurr= randomJoints.row(i).transpose();
////
////        // Positions
////        positions.row(i) = robot.getCurrentPosition(jointsCurr);
////
////        // Compute manipulabilities
////        JFull = robot.getPoseJacobian(jointsCurr);
////        Jgeo = robot.buildGeometricJacobian(JFull, jointsCurr);
////        M=Jgeo.bottomRows(3)*Jgeo.bottomRows(3).transpose();
////        Mresized=M;
////        Mresized.resize(1,9);
////        manips.row(i) = Mresized;
//
//            M = manips.row(i);
//            M.resize(3, 3);
//
//            // Normalize manipulabilities to volume = 1
//            double vol = getEllipsoidVolume(M);
//            Mnormalized = scaleEllipsoidVolume(M, 1 / vol);
//            assert(getEllipsoidVolume(Mnormalized) - 1 < 1e-4);
//            Mnormalized.resize(1, 9);
//            manipsNormalized.row(i) = Mnormalized;
//            scales(i, 0) = vol;
//        }
//
//        // Normalize scales
//        scalesNormalized = (scales.array() - scales.minCoeff()) / (scales.maxCoeff() - scales.minCoeff());
//        assert(scalesNormalized.minCoeff() >= 0 && scalesNormalized.maxCoeff() <= 1);
//
//        deb(manipsNormalized.topRows(5))
//        deb(manips.topRows(5))
//        deb(scales.topRows(5))
//
//        writeCSV(manipsNormalized, manips_normalized_path);
//        writeCSV(manips, manips_path);
//        writeCSV(scales, scales_path);
//        writeCSV(scalesNormalized, scales_normalized_path);
//

}

/*
 * // calibration process with controlling towards unitsphere
int main() {


int n=1; // number of random configs used in calibration process
string exp="cut_userchoice"; // experiment name
string proband="4"; // user number

string infoPath="/home/nnrthmr/Desktop/RHuMAn-arm-model/data/"+exp+"/agg/"+proband+"/info.txt";
string fileXd = "/home/nnrthmr/CLionProjects/ma_thesis/data/learning/rhuman/" + exp + "/" + proband + "/xd.csv";
string fileXhat = "/home/nnrthmr/CLionProjects/ma_thesis/data/learning/rhuman/" + exp + "/" + proband + "/xhat.csv";

string fileBaseOutput = "/home/nnrthmr/CLionProjects/ma_thesis/data/tracking/rhuman";
string fileErrors = fileBaseOutput + "/" + exp + "/" + proband +"/errorManipulabilities.csv";
string fileControlledManipulabilities = fileBaseOutput + "/" + exp + "/" + proband +"/controlledManipulabilities.csv";

string fileRobotScales = "/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/finalScalesRobot.csv";
string fileRobotPos = "/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/finalPosRobot.csv";
string fileHumanScales = "/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/finalScalesHuman.csv";
string fileHumanPos = "/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/finalPosHuman.csv";

Franka robot = Franka(false);
MatrixXd pos(3,n);
MatrixXd scales(1,n);

//Calibration part for robot arm
//if(!fileExists(fileRobotScales) || !fileExists(fileRobotPos)){
deb("ROBOT")
    pos.setZero();
    scales.setZero();
    calibrationProcessRobot(robot, pos, scales);
    writeCSV(scales, fileRobotScales);
    writeCSV(pos, fileRobotPos);
//}


//Calibration part for human arm
if(!fileExists(fileHumanPos) || !fileExists(fileHumanScales)){
    pos.setZero();
    scales.setZero();
    calibrationProcessHuman(pos, scales, 1.35);
    writeCSV(scales, fileHumanScales);
    writeCSV(pos, fileHumanPos);
}


// Read meta information on experiment specified above
ifstream infile(infoPath);
int nPoints, nDemos, totalPoints;
infile >> nPoints >> nDemos >> totalPoints;
assert(nPoints*nDemos==totalPoints);


// Precompute scales at learned positions in trajectory
MatrixXd xd(3,nPoints);
MatrixXd xhat(nPoints,9);
MatrixXd ratios(1,nPoints);
if(!fileExists(fileXd) || !fileExists(fileXhat)) {
    learn3dRHumanMotion(xd, xhat, nPoints, nDemos, totalPoints, exp, proband);
}
else {
    loadCSV(fileXd, &xd);
    loadCSV(fileXhat, &xhat);
}

precomputeScalingRatios(robot, xd, ratios);

// Control robot with scaled manipulabilities from learned trajectory/manipulabilities
MatrixXd controlledMatrix(xd.cols(), 9);
MatrixXd errMatrix(xd.cols(),1);
controlManipulabilitiesRHumanArm(robot, xd, xhat, nPoints, true, ratios, errMatrix, controlledMatrix);


// Write results
if (mkdir((fileBaseOutput + "/" + exp).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1)
{
    if( errno == EEXIST ) {
    } else {
        throw std::runtime_error( strerror(errno) );
    }
}
if (mkdir((fileBaseOutput + "/" + exp+"/"+proband).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1)
{
    if( errno == EEXIST ) {
    } else {
        throw std::runtime_error( strerror(errno) );
    }
}

writeCSV(errMatrix, fileErrors);
writeCSV(controlledMatrix, fileControlledManipulabilities);


// Plotting in MATLAB (interpolation plots)
unique_ptr<MATLABEngine> matlabPtr = startMATLAB();
matlab::data::ArrayFactory factory;
matlab::data::TypedArray<int>  args_type = factory.createScalar<int>(0);
matlabPtr->setVariable(u"type_m", std::move(args_type));
matlabPtr->eval(u"plotInterpolation(type_m);");

// Plotting in Python (manipulabilities)
//char plottingFile[] = "/home/nnrthmr/CLionProjects/ma_thesis/py/plot_calibration_process.py";
//runPythonScript(plottingFile);

}

 */

