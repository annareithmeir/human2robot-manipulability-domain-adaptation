#ifndef MA_THESIS_CONTROL_H
#define MA_THESIS_CONTROL_H

#include "Franka.h"
#include "Mapping_utils.h"

MatrixXd manipulabilityTrackingMainTask(Franka &robot, const MatrixXd& MDesired, vector<MatrixXd> &mLoop, vector<double> &eLoop);
MatrixXd manipulabilityTrackingNullspace(Franka &robot, const MatrixXd& MDesired, vector<MatrixXd> &mLoop, vector<double> &eLoop);
MatrixXd manipulabilityTrackingSecondaryTask(Franka robot, const MatrixXd& XDesired, const MatrixXd& DXDesired, const MatrixXd& MDesired);
void controlManipulabilitiesRHumanArm(Franka &robot, MatrixXd &xd, MatrixXd &xHat, int nPoints, bool mainTask, MatrixXd &ratios, MatrixXd &errors, MatrixXd &controlledManips);
void unitShpereTrackingMainTask(Franka robot, const MatrixXd& PosInit, vector<MatrixXd> &finalM, vector<MatrixXd> &finalPos, vector<MatrixXd> &mLoop, vector<double> &eLoop);
void precomputeScalingRatios(Franka &robot, MatrixXd &xd, MatrixXd &ratios);
void calibrationProcessRobot(Franka robot, MatrixXd &positions, MatrixXd &scales);
void calibrationProcessHuman(MatrixXd &positions, MatrixXd &scales, double shoulderHeight);


#endif //MA_THESIS_CONTROL_H
