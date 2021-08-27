#ifndef MA_THESIS_SPD_H
#define MA_THESIS_SPD_H

#include <Eigen/Dense>
using Eigen::MatrixXd;

class SPD {
public:
    SPD exp();
    SPD log();

    SPD normal(SPD x, SPD mu, MatrixXd sigma);
};


#endif //MA_THESIS_SPD_H
