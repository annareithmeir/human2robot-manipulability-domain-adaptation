#include <Franka.h>
#include <GMM.h>
#include <GMM_SPD.h>
#include <control.h>
#include <learn.h>
#include <errno.h>
#include <sys/stat.h>

using namespace std;
using namespace Eigen;
//#define dimensions 3

int main() {


int n=30; // number of random configs used in calibration process
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
if(!fileExists(fileRobotScales) || !fileExists(fileRobotPos)){
    pos.setZero();
    scales.setZero();
    calibrationProcessRobot(robot, pos, scales);
    writeCSV(scales, fileRobotScales);
    writeCSV(pos, fileRobotPos);
}


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

