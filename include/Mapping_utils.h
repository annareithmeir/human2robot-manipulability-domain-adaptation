#ifndef MA_THESIS_MAPPING_UTILS_H
#define MA_THESIS_MAPPING_UTILS_H

#include "utils.h"
#include <Eigen/Eigenvalues>
#include "MatlabEngine.hpp"
#include "MatlabDataArray.hpp"

using namespace std;
using namespace Eigen;
using namespace matlab::engine;
using namespace matlab::data;

// Checked!
inline
double getInterpolatedPoint(const MatrixXd pos, const MatrixXd scales, double x, double y, double z, int type) {
    unique_ptr<MATLABEngine> matlabPtr = startMATLAB();
    matlab::data::ArrayFactory factory;

    const unsigned long posRows=pos.rows();
    const unsigned long posCols=pos.cols();

    TypedArray<double> pos_matlab = factory.createArray<double>({ posRows, posCols }, pos.data(), pos.data()+posRows*posCols);
    TypedArray<double> scales_matlab = factory.createArray<double>({ 1, posCols }, scales.data(), scales.data()+posCols);

    TypedArray<double>  args_x = factory.createScalar<double>(x);
    TypedArray<double>  args_y = factory.createScalar<double>(y);
    matlab::data::TypedArray<double>  args_z = factory.createScalar<double>(z);
    matlab::data::TypedArray<double>  args_type = factory.createScalar<double>(type);

    matlabPtr->setVariable(u"pos_m", std::move(pos_matlab));
    matlabPtr->setVariable(u"scales_m", std::move(scales_matlab));
    matlabPtr->setVariable(u"x_m", std::move(args_x));
    matlabPtr->setVariable(u"y_m", std::move(args_y));
    matlabPtr->setVariable(u"z_m", std::move(args_z));
    matlabPtr->setVariable(u"type_m", std::move(args_type));

    matlabPtr->eval(u"v=interpolationAtPoint(pos_m,scales_m,x_m,y_m,z_m,type_m)");
    matlab::data::TypedArray<double> const v = matlabPtr->getVariable(u"v");

    return v[0];
}

inline
void getInterpolatedPoints(const MatrixXd pos, const MatrixXd scales, MatrixXd &xd, int type, MatrixXd &scalesInterpolated) {
    unique_ptr<MATLABEngine> matlabPtr = startMATLAB();
    matlab::data::ArrayFactory factory;

    const unsigned long posRows=pos.rows();
    const unsigned long posCols=pos.cols();

    TypedArray<double> pos_matlab = factory.createArray<double>({ posRows, posCols }, pos.data(), pos.data()+posRows*posCols);
    TypedArray<double> scales_matlab = factory.createArray<double>({ 1, posCols }, scales.data(), scales.data()+posCols);
    for(int i=0;i<xd.cols();++i){
        TypedArray<double>  args_x = factory.createScalar<double>(xd(0,i));
        TypedArray<double>  args_y = factory.createScalar<double>(xd(1,i));
        matlab::data::TypedArray<double>  args_z = factory.createScalar<double>(xd(2,i));
        matlab::data::TypedArray<double>  args_type = factory.createScalar<double>(type);

        matlabPtr->setVariable(u"pos_m", std::move(pos_matlab));
        matlabPtr->setVariable(u"scales_m", std::move(scales_matlab));
        matlabPtr->setVariable(u"x_m", std::move(args_x));
        matlabPtr->setVariable(u"y_m", std::move(args_y));
        matlabPtr->setVariable(u"z_m", std::move(args_z));
        matlabPtr->setVariable(u"type_m", std::move(args_type));

        matlabPtr->eval(u"v=interpolationAtPoint(pos_m,scales_m,x_m,y_m,z_m,type_m)");
        matlab::data::TypedArray<double> v = matlabPtr->getVariable(u"v");
        scalesInterpolated(0,i) = v[0];
    }
}

inline
double getScalingRatioAtPoint(int num, double x, double y, double z){
    int interpolationType=0;
    MatrixXd posHuman(3, num);
    MatrixXd scalesHuman(1, num);
    MatrixXd posRobot(3, num);
    MatrixXd scalesRobot(1, num);

    loadCSV("/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/finalPosUser1.csv", &posHuman);
    loadCSV("/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/finalScalesUser1.csv", &scalesHuman);
    loadCSV("/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/finalPosRobot.csv", &posRobot);
    loadCSV("/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/finalScalesRobot.csv", &scalesRobot);
    double robotInterpolatedPoint = getInterpolatedPoint(posRobot, scalesRobot, x,y,z, interpolationType);
    double humanInterpolatedPoint = getInterpolatedPoint(posHuman, scalesHuman, x,y,z, interpolationType);
    return robotInterpolatedPoint/humanInterpolatedPoint;
}

inline
void getScalingRatiosAtPoints(int num, MatrixXd &xd, MatrixXd &ratios){
    int interpolationType=0;

    MatrixXd scalesInterpolatedHuman(1, xd.cols());
    MatrixXd scalesInterpolatedRobot(1, xd.cols());

    MatrixXd posHuman(3, num);
    MatrixXd scalesHuman(1, num);
    MatrixXd posRobot(3, num);
    MatrixXd scalesRobot(1, num);

    assert(fileExists("/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/finalPosHuman.csv") &&
                   fileExists("/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/finalScalesHuman.csv") &&
                   fileExists("/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/finalPosRobot.csv") &&
                   fileExists("/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/finalScalesRobot.csv"));

    loadCSV("/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/finalPosHuman.csv", &posHuman);
    loadCSV("/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/finalScalesHuman.csv", &scalesHuman);
    loadCSV("/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/finalPosRobot.csv", &posRobot);
    loadCSV("/home/nnrthmr/CLionProjects/ma_thesis/data/calibration/finalScalesRobot.csv", &scalesRobot);

    getInterpolatedPoints(posRobot, scalesRobot, xd, interpolationType, scalesInterpolatedRobot);
    getInterpolatedPoints(posHuman, scalesHuman, xd, interpolationType, scalesInterpolatedHuman);
    ratios = (scalesInterpolatedRobot.array()/scalesInterpolatedHuman.array()).matrix();
}

inline
double getScalingRatioAtPoint(const MatrixXd posRobot, const MatrixXd scalesRobot, const MatrixXd posHuman, const MatrixXd scalesHuman, double x, double y, double z){
    int interpolationType=0;
    double robotInterpolatedPoint = getInterpolatedPoint(posRobot, scalesRobot, x,y,z, interpolationType);
    double humanInterpolatedPoint = getInterpolatedPoint(posHuman, scalesHuman, x,y,z, interpolationType);
    return robotInterpolatedPoint/humanInterpolatedPoint;
}

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

// as described page 1 in paper
inline
double getLengthOfPrincipalAxis(MatrixXd ellipsoid){
    EigenSolver<MatrixXd> es(ellipsoid);
    MatrixXd eigvals= es.eigenvalues().real();
    return sqrt(eigvals.maxCoeff());
}

inline
MatrixXd getRatiosOfAxesLengths(MatrixXd ellipsoidTeacher, MatrixXd ellipsoidStudent){ //
    MatrixXd lengthTeacher = getLengthsOfPrincipalAxes(ellipsoidTeacher);
    MatrixXd lengthStudent = getLengthsOfPrincipalAxes(ellipsoidStudent);
    return (lengthStudent.array()/lengthTeacher.array()).matrix();
}



#endif //MA_THESIS_MAPPING_UTILS_H
