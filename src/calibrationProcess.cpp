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

void printProgress(double percentage) {
    int val = (int) (percentage * 100);
    int lpad = (int) (percentage * PBWIDTH);
    int rpad = PBWIDTH - lpad;
    printf("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
    fflush(stdout);
}

void generate_human_robot_data_random(int num, double shoulder_height){

    string base_path = "/home/nnrthmr/testing";
//    string base_path = "/home/nnrthmr/PycharmProjects/ma_thesis/data";

    string manips_normalized_path=base_path+"/r_manipulabilities_normalized.csv";
    string manips_path=base_path+"/r_manipulabilities.csv";
    string scales_path=base_path+"/r_scales.csv";
    string positions_path=base_path+"/r_positions.csv";
    string scales_normalized_path=base_path+"/r_scales_normalized.csv";

    Franka robot = Franka(false);
    MatrixXd positions(num, 3);
    MatrixXd scales(num,1);
    MatrixXd scalesNormalized(num,1);
    MatrixXd manipsNormalized(num,9);
    MatrixXd manips(num,9);

    MatrixXd JFull, Jgeo, M, Mnormalized, Mresized;
    MatrixXd randomJoints = robot.GetRandomJointConfig(num); // num x 7
    VectorXd jointsCurr(7);

    cout<< " Generating robot data ..."<<endl;

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

//    deb(manipsNormalized.topRows(5))
//    deb(manips.topRows(5))
//    deb(scales.topRows(5))

    writeCSV(manipsNormalized, manips_normalized_path);
    writeCSV(manips, manips_path);
    writeCSV(scales, scales_path);
    writeCSV(positions, positions_path);
    writeCSV(scalesNormalized, scales_normalized_path);


    cout<< " Generating human data ..."<<endl;

    unique_ptr<MATLABEngine> matlabPtr = startMATLAB();
    matlab::data::ArrayFactory factory;
    matlab::data::TypedArray<float>  args_shoulder_height = factory.createScalar<float>(shoulder_height);
    matlab::data::TypedArray<int>  args_num = factory.createScalar<int>(num);
    matlab::data::CharArray args_base_path = factory.createCharArray(base_path);
    matlabPtr->setVariable(u"shoulder_height_m", std::move(args_shoulder_height));
    matlabPtr->setVariable(u"num_m", std::move(args_num));
    matlabPtr->setVariable(u"base_path_m", std::move(args_base_path));
    matlabPtr->eval(u"generateHumanData(shoulder_height_m, num_m, base_path_m);");
    matlabPtr->eval(u"createLookupTable(base_path_m);");

}

void mapManipulabilities(){
    string base_path = "/home/nnrthmr/testing";
    unique_ptr<MATLABEngine> matlabPtr = startMATLAB();
    matlab::data::ArrayFactory factory;
    matlab::data::CharArray args_base_path = factory.createCharArray(base_path);
    matlabPtr->setVariable(u"base_path_m", std::move(args_base_path));
    matlabPtr->eval(u"map_manipulabilities(base_path_m);");
}

int main(){

    int a= 1;
    if (a==1) generate_human_robot_data_random(10, 1.35);
    else {
        string manips_normalized_path = "/home/nnrthmr/PycharmProjects/ma_thesis/data/r_manipulabilities_normalized.csv";
        string manips_path = "/home/nnrthmr/PycharmProjects/ma_thesis/data/r_manipulabilities.csv";
        string scales_path = "/home/nnrthmr/PycharmProjects/ma_thesis/data/r_scales.csv";
        string scales_normalized_path = "/home/nnrthmr/PycharmProjects/ma_thesis/data/r_scales_normalized.csv";

//    string manips_normalized_path="/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/affineTrafo/r_manipulabilities_normalized.csv";
//    string positions_path="/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/affineTrafo/r_positions.csv";
//    string scales_normalized_path="/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/affineTrafo/r_scales_normalized.csv";

        string robot_manips_path = "/home/nnrthmr/PycharmProjects/ma_thesis/data/r_manipulabilities.csv";
        MatrixXd manips(500, 9);
        loadCSV(robot_manips_path, &manips);
        int num = 500;
        Franka robot = Franka(false);
        MatrixXd positions(num, 3);
        MatrixXd scales(num, 1);
        MatrixXd scalesNormalized(num, 1);
        MatrixXd manipsNormalized(num, 9);
//    MatrixXd manips(num,9);

        MatrixXd JFull, Jgeo, M, Mnormalized, Mresized;
//    MatrixXd randomJoints = robot.GetRandomJointConfig(num); // num x 7
        VectorXd jointsCurr(7);

        for (int i = 0; i < num; ++i) {
            deb(i)

//        jointsCurr= randomJoints.row(i).transpose();
//
//        // Positions
//        positions.row(i) = robot.getCurrentPosition(jointsCurr);
//
//        // Compute manipulabilities
//        JFull = robot.getPoseJacobian(jointsCurr);
//        Jgeo = robot.buildGeometricJacobian(JFull, jointsCurr);
//        M=Jgeo.bottomRows(3)*Jgeo.bottomRows(3).transpose();
//        Mresized=M;
//        Mresized.resize(1,9);
//        manips.row(i) = Mresized;

            M = manips.row(i);
            M.resize(3, 3);

            // Normalize manipulabilities to volume = 1
            double vol = getEllipsoidVolume(M);
            Mnormalized = scaleEllipsoidVolume(M, 1 / vol);
            assert(getEllipsoidVolume(Mnormalized) - 1 < 1e-4);
            Mnormalized.resize(1, 9);
            manipsNormalized.row(i) = Mnormalized;
            scales(i, 0) = vol;
        }

        // Normalize scales
        scalesNormalized = (scales.array() - scales.minCoeff()) / (scales.maxCoeff() - scales.minCoeff());
        assert(scalesNormalized.minCoeff() >= 0 && scalesNormalized.maxCoeff() <= 1);

        deb(manipsNormalized.topRows(5))
        deb(manips.topRows(5))
        deb(scales.topRows(5))

        writeCSV(manipsNormalized, manips_normalized_path);
        writeCSV(manips, manips_path);
        writeCSV(scales, scales_path);
        writeCSV(scalesNormalized, scales_normalized_path);
    }

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

