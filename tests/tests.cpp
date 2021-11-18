
#include <utils.h>
#include <Franka.h>
#include <Mapping_utils.h>

using namespace std;
#define CATCH_CONFIG_MAIN
#include "catch2.hpp"


TEST_CASE("simpleTest", ""){
    int a=2;
    REQUIRE(a==2);
}

TEST_CASE("Ellipsoid principal axes"){
    MatrixXd A(3,3);
    A << 1,0,0,0,1,0,0,0,1;
    MatrixXd a1=getPrincipalAxes(A);
    MatrixXd a2=getLengthsOfPrincipalAxes(A);

    REQUIRE(a1==A);
    REQUIRE(a2.minCoeff()==1);
    REQUIRE(a2.maxCoeff()==1);

}

TEST_CASE("Ellipsoid axes ratio"){
    MatrixXd T(3,3);
    T << 1,0,0,0,1,0,0,0,1;
    MatrixXd S(3,3);
    S << 3,0,0,0,2,0,0,0,0.5;
    MatrixXd a1=getRatiosOfAxesLengths(T,S);
    MatrixXd resultDesired(1,3);
    resultDesired << 3,2,0.5;

    REQUIRE(a1==resultDesired);

}

TEST_CASE("Ellipsoid alength of principal axis"){
    MatrixXd T(3,3);
    T << 1,0,0,0,1,0,0,0,1;
    double length = getLengthOfPrincipalAxis(T);

    REQUIRE(length==1.0);

}

TEST_CASE("Matrix-Vector multiplication"){
    MatrixXd m(3,3);
    MatrixXd result(3,3);
    MatrixXd v(1,3);
    m.setIdentity();
    v << 1,2,3;
    result << 1,0,0,0,2,0,0,0,3;

    MatrixXd m3=(m* v.transpose()).asDiagonal();

    REQUIRE(m3==result);
}

TEST_CASE("MATLAB interpolation function"){
    MatrixXd pos(3,3);
    pos.setIdentity();
    MatrixXd scales(1,3);
    scales.setOnes();
    double x = getInterpolatedPoint(pos, scales,0,0,0,0);
}

TEST_CASE("Franka random generator"){
    Franka franka;
    MatrixXd m = franka.GetJointConstraints();
    MatrixXd m2 = franka.GetRandomJointConfig(1);
    
    REQUIRE(m2.rows()==1);
    REQUIRE(m.col(0).minCoeff() <= m2(0,0));
    REQUIRE(m2(0,0) <= m.col(0).maxCoeff());
    REQUIRE(m.col(1).minCoeff() <= m2(0,1));
    REQUIRE(m2(0,1) <= m.col(1).maxCoeff());
    REQUIRE(m.col(2).minCoeff() <= m2(0,2));
    REQUIRE(m2(0,2) <= m.col(2).maxCoeff());
    REQUIRE(m.col(3).minCoeff() <= m2(0,3));
    REQUIRE(m2(0,3) <= m.col(3).maxCoeff());
    REQUIRE(m.col(4).minCoeff() <= m2(0,4));
    REQUIRE(m2(0,4) <= m.col(4).maxCoeff());
    REQUIRE(m.col(5).minCoeff() <= m2(0,5));
    REQUIRE(m2(0,5) <= m.col(5).maxCoeff());
    REQUIRE(m.col(6).minCoeff() <= m2(0,6));
    REQUIRE(m2(0,6) <= m.col(6).maxCoeff());
}

TEST_CASE("buildGeometricJacobian", ""){
    MatrixXd m(8,7);
    // Jacobian of initial position of franka_kinematic scene
    m << 0.118311,   0.205096 ,-0.0532549,  -0.313965 , -0.379302,  -0.313965 ,  0.118311,
         -0.313965 ,  -0.11088 , -0.321125,   0.118311,  0.0131348 ,  0.118311 ,  0.313965,
         0.364292, -0.0801406 ,  0.379302 ,  0.068698 , -0.053255 ,  0.068698 , -0.364292,
         0.068698,  -0.434993, -0.0131348 ,  0.364292  ,-0.321125 ,  0.364292 ,  0.068698,
         0.0762674 ,  0.019253 , 0.0766116,  -0.137679  , 0.116601 , -0.148522 , 0.0761733,
         -0.0990959 ,  0.077098,  -0.145394, -0.0685005 , -0.119273 ,  0.055895 , 0.0991314,
         -0.0933655,  -0.142853 , -0.115448 ,  0.080415 ,-0.0996139, -0.0696307,  0.0933861,
         -0.0891384 , 0.0157437 ,  -0.08986  ,-0.111577 , -0.126085,  -0.133026 ,-0.0890291;

    DQ x = DQ(0.137396, 0.728585, 0.62793,  - 0.236622, - 0.178277 ,- 0.186731,0.198192, - 0.152535);
    MatrixXd qt(7,1);
    qt <<1.15192,
        0.383972,
        0.261799,
        -1.5708,
        0,
        1.39626,
        0;

    Franka robot = Franka();
    MatrixXd result = robot.buildGeometricJacobian(m, qt);
    deb(result)

    // result of matlab code luis gave me
    MatrixXd matlabResult(6,7);
    matlabResult <<   -2.36621999995357e-07,	-0.913545042366400,	0.152365787393000,	0.980021962794000,	0.127826181581600,	0.980021962794000,	-0.172248102558000,
                      -1.37396000000900e-07,	0.406738186214800,	0.342219969914000,	-0.173651598808000,	0.923433912718800,	-0.173651598808000,	-0.497373296844000,
                      0.999999090240000,	-8.66334000003410e-07,	0.927183637982800,	-0.0969549129960000,	-0.361842034966000,	-0.0969549129960000,	-0.850263826840000,
                      -1.17798165856160,	0.0520821316760000,	-1.04838568899000,	0.147620364420000,	-0.236592140248800,	0.0516959927303999	,0.000587960852399982,
                      0.221396070307200,	0.116976336281200,	0.185765006996000,	0.380189634660000,	0.0419247346824000,	0.269732329287200,	-0.000104052462799931,
                      -6.64324400045935e-07,	-1.16619089653360,	0.103719191956400,	0.811225776076000,	0.0234046821684000,	0.0394441288780001	,-5.75407019999807e-05;
//    deb((result.array()-matlabResult.array()).maxCoeff())
//    deb((result.array()-matlabResult.array()).matrix().norm());
    REQUIRE((result.array()-matlabResult.array()).maxCoeff() < 1e-5);

//                        4.92605e-07    -0.913546     0.152366     0.980022     0.127824     0.980022    -0.172248
//                        3.16379e-06     0.406736     0.342223     -0.17365     0.923433     -0.17365    -0.497376
//                        0.999999 -1.54287e-06     0.927182   -0.0969551    -0.361845   -0.0969551    -0.850262
//                        -1.17798    0.0520813     -1.04839     0.147619    -0.236594    0.0516946  0.000587347
//                        0.221394     0.116975     0.185763      0.38019    0.0419246      0.26973 -0.000104173
//                        8.53882e-07     -1.16619     0.103721      0.81122    0.0234063    0.0394387 -5.94086e-05
}


