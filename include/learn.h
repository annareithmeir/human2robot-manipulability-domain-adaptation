#ifndef MA_THESIS_LEARN_H
#define MA_THESIS_LEARN_H

#include <utils.h>
#include <GMM_SPD.h>
#include <GMM.h>

void learn2d();
void learn3d();
void learn3dHumanMotion(MatrixXd &xd, MatrixXd &xHat);
void learn3dRHumanMotion(MatrixXd &xd, MatrixXd &xHat, const int nPoints, const int nDemos, const int totalPoints, const string exp, const string proband);

#endif //MA_THESIS_LEARN_H
