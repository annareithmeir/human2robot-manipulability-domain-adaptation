#include <Franka.h>
#include <GMM.h>
#include <GMM_SPD.h>
#include <control.h>
#include <learn.h>
#include <errno.h>
#include <sys/stat.h>

using namespace std;
using namespace Eigen;

#define dimensions 3

int main() {

int n=30;
MatrixXd pos(3,n);
MatrixXd scales(1,n);

calibrationProcessHuman(pos, scales);
    writeCSV(scales, "/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/finalScalesUser1.csv");
    writeCSV(pos, "/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/finalPosUser1.csv");

pos.setZero();
scales.setZero();
Franka robot = Franka(false);
calibrationProcessRobot(robot, pos, scales);
    writeCSV(scales, "/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/finalScalesRobot.csv");
    writeCSV(pos, "/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/finalPosRobot.csv");

//loadCSV("/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/finalScalesRobot.csv", &scales);
//loadCSV("/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/finalPosRobot.csv", &pos);
//double x = getInterpolatedPoint(pos, scales,0.5,0.5,0.5,0);
//double x= getScalingRatioAtPoint(pos, scales,0.5,0.5,0.5);
//deb(x)

//Control part
string exp="cut_userchoice";
string proband="4";

string infoPath;
if (proband=="") infoPath = "/home/nnrthmr/Desktop/RHuMAn-arm-model/data/"+exp+"/agg/info.txt";
else infoPath="/home/nnrthmr/Desktop/RHuMAn-arm-model/data/"+exp+"/agg/"+proband+"/info.txt";

ifstream infile(infoPath);
int nPoints, nDemos, totalPoints;
infile >> nPoints >> nDemos >> totalPoints;
assert(nPoints*nDemos==totalPoints);

MatrixXd xd(3,nPoints);
MatrixXd xHat(6,nPoints);
MatrixXd ratios(1,nPoints);
if(!fileExists("/home/nnrthmr/CLionProjects/ma_thesis/data/results/rhuman/" + exp + "/" + proband + "/xd.csv" )
    || !fileExists("/home/nnrthmr/CLionProjects/ma_thesis/data/results/rhuman/" + exp + "/" + proband + "/xhat.csv"))
        learn3dRHumanMotion(xd, xHat, nPoints, nDemos, totalPoints, exp, proband);
precomputeScalingRatios(robot, xd, ratios);
controlManipulabilitiesRHumanArm(robot, exp, proband, nPoints, ratios);
}

