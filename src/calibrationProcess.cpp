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
string exp="cut_userchoice";
string proband="4";

//Calibration part
MatrixXd pos(3,n);
MatrixXd scales(1,n);
calibrationProcessHuman(pos, scales, 1.35);
writeCSV(scales, "/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/finalScalesUser1.csv");
writeCSV(pos, "/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/finalPosUser1.csv");

pos.setZero();
scales.setZero();
Franka robot = Franka(false);
calibrationProcessRobot(robot, pos, scales);
writeCSV(scales, "/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/finalScalesRobot.csv");
writeCSV(pos, "/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/finalPosRobot.csv");

string infoPath;
if (proband=="") infoPath = "/home/nnrthmr/Desktop/RHuMAn-arm-model/data/"+exp+"/agg/info.txt";
else infoPath="/home/nnrthmr/Desktop/RHuMAn-arm-model/data/"+exp+"/agg/"+proband+"/info.txt";

ifstream infile(infoPath);
int nPoints, nDemos, totalPoints;
infile >> nPoints >> nDemos >> totalPoints;
assert(nPoints*nDemos==totalPoints);

// Precompute scales at xd
MatrixXd xd(3,nPoints);
MatrixXd xhat(6,nPoints);
MatrixXd ratios(1,nPoints);
if(!fileExists("/home/nnrthmr/CLionProjects/ma_thesis/data/results/rhuman/" + exp + "/" + proband + "/xd.csv" )
    || !fileExists("/home/nnrthmr/CLionProjects/ma_thesis/data/results/rhuman/" + exp + "/" + proband + "/xhat.csv"))
        learn3dRHumanMotion(xd, xhat, nPoints, nDemos, totalPoints, exp, proband);
else{
//    loadCSV("/home/nnrthmr/CLionProjects/ma_thesis/data/results/rhuman/" + exp + "/" + proband + "/xhat.csv", &xhat);
    loadCSV("/home/nnrthmr/CLionProjects/ma_thesis/data/results/rhuman/" + exp + "/" + proband + "/xd.csv", &xd);
}

precomputeScalingRatios(robot, xd, ratios);

// Control part with scaled manipulabilities
//controlManipulabilitiesRHumanArm(robot, exp, proband, nPoints, true, ratios);
}

