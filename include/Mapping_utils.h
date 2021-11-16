#ifndef MA_THESIS_MAPPING_UTILS_H
#define MA_THESIS_MAPPING_UTILS_H

#include "utils.h"
#include <Eigen/Eigenvalues>

using namespace std;
using namespace Eigen;

inline
MatrixXd getPrincipalAxes(MatrixXd ellipsoid){ // colwise
    EigenSolver<MatrixXd> es(ellipsoid);
    MatrixXd eigs= es.eigenvectors().real();
    return eigs;
}

inline
MatrixXd getEigenValues(MatrixXd ellipsoid){ // colwise
    EigenSolver<MatrixXd> es(ellipsoid);
    MatrixXd eigs= es.eigenvalues().real();
    return eigs;
}

inline
MatrixXd getSingularValues(MatrixXd ellipsoid){ // colwise
    JacobiSVD<MatrixXd> svd(ellipsoid, ComputeThinU | ComputeThinV);
    MatrixXd sings = svd.singularValues().real();
    return sings;
}


inline
MatrixXd getLengthsOfPrincipalAxes(MatrixXd ellipsoid){
    EigenSolver<MatrixXd> es(ellipsoid);
    MatrixXd eigvals= es.eigenvalues().real();
    eigvals=eigvals.array().sqrt().pow(-1);
    return eigvals;
}

inline
MatrixXd getRatiosOfAxesLengths(MatrixXd ellipsoidTeacher, MatrixXd ellipsoidStudent){ //
    MatrixXd lengthTeacher = getLengthsOfPrincipalAxes(ellipsoidTeacher);
    MatrixXd lengthStudent = getLengthsOfPrincipalAxes(ellipsoidStudent);
    return (lengthStudent.array()/lengthTeacher.array()).matrix();
}


#endif //MA_THESIS_MAPPING_UTILS_H
