#include <catch2.hpp>
#include <utils.h>

using namespace std;
using namespace Eigen;
#define CONFIG_CATCH_MAIN

/*
TEST_CASE("tensor_center()", ""){
    Eigen::array<int, 2> dims({1,2});
    Eigen::Tensor<float, 0> z;
    z.setZero();
    Tensor<double, 3> x(2,2,2);
    x.setRandom();
    REQUIRE(tensor_center(x).mean(dims).all() == z);
}
 */

