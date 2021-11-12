#ifndef MA_THESIS_MAPPING_UTILS_H
#define MA_THESIS_MAPPING_UTILS_H

#include "utils.h"
#include <Eigen/Eigenvalues>

using namespace std;
using namespace Eigen;


MatrixXd getPrincipalAxes(MatrixXd ellipsoid){ // colwise
    EigenSolver<MatrixXd> es(ellipsoid);
    MatrixXd eigs= es.eigenvectors().real();
    return eigs;
}

MatrixXd getLengthsOfPrincipalAxes(MatrixXd ellipsoid){
    EigenSolver<MatrixXd> es(ellipsoid);
    MatrixXd eigvals= es.eigenvalues().real();
    eigvals=eigvals.array().sqrt().pow(-1);
    return eigvals;
}

MatrixXd getRatiosOfAxesLengths(MatrixXd ellipsoidTeacher, MatrixXd ellipsoidStudent){ //
    MatrixXd lengthTeacher = getLengthsOfPrincipalAxes(ellipsoidTeacher);
    MatrixXd lengthStudent = getLengthsOfPrincipalAxes(ellipsoidStudent);
    return (lengthStudent.array()/lengthTeacher.array()).matrix();
}


#endif //MA_THESIS_MAPPING_UTILS_H
