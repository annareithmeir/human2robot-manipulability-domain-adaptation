#include <Franka.h>
#include <GMM.h>
#include <GMM_SPD.h>

using namespace std;
using namespace Eigen;

#define deb(x) cout << #x << " " << x << endl;
#define dimensions 3

int main() {
    // Load the demonstration data
//    string data_path="/home/nnrthmr/Desktop/master-thesis/vrep/vrep_franka_promps/py_scripts/data/";
//    MatrixXd data_pos(4, 2*21);
//    vector<Tensor3d> data_m;
//    load_data(data_path, 2, 21, &data_m, &data_pos);

//    std::cout.precision(20);


    // Build GMM for trajectories in cartesian coordinates

    if(dimensions==2){
        string cmat_path="/home/nnrthmr/CLionProjects/ma_thesis/data/C_Mat.csv";
        //MatrixXd data(3, 400);
        MatrixXd data(4, 400);
        vector<Tensor3d> data_m;
        load_data_cmat(cmat_path, &data);
        data.row(3)=data.row(2)*0.75;

        GMM model = GMM();
        model.InitModel(data);
        model.TrainEM();

        MatrixXd expData(3, 100);
//    MatrixXd expData(2, 100);
        expData.setZero();
        std::vector<MatrixXd> expSigma; //m_dimOut x m_dimOut x m_nData
        std::cout<<"GMR"<<std::endl;
        model.GMR(expData, expSigma);

        WriteCSV(expData, "/home/nnrthmr/CLionProjects/ma_thesis/data/expData3d.csv");
        WriteCSV(expSigma, "/home/nnrthmr/CLionProjects/ma_thesis/data/expSigma3d.csv");
    }
    else{
        string cmat_path="/home/nnrthmr/CLionProjects/ma_thesis/data/demos/trajectories.csv";
        MatrixXd data(80, 4);
        load_data_mmat(cmat_path, &data);
        deb(data);

        GMM model = GMM();
        model.InitModel(data.transpose());
        model.TrainEM();

        MatrixXd expData(3, 20);
        expData.setZero();
        std::vector<MatrixXd> expSigma; //m_dimOut x m_dimOut x m_nData
        std::cout<<"GMR"<<std::endl;
        model.GMR(expData, expSigma);

        WriteCSV(expData, "/home/nnrthmr/CLionProjects/ma_thesis/data/expData3d.csv");
        WriteCSV(expSigma, "/home/nnrthmr/CLionProjects/ma_thesis/data/expSigma3d.csv");
    }



//     Build GMM for Manifold
if(dimensions==2){
    string mmat_path="/home/nnrthmr/CLionProjects/ma_thesis/data/Manip_Mat.csv";
    MatrixXd data(400, 4);
    load_data_mmat_skip_first(mmat_path, &data);

    GMM_SPD model2 = GMM_SPD();
    model2.InitModel(data);
    model2.TrainEM();

    MatrixXd expData(3, 100);
    expData.setZero();
    std::vector<MatrixXd> expSigma; //m_dimOut x m_dimOut x m_nData

    std::cout<<"GMR"<<std::endl;
    model2.GMR(expData, expSigma);
}
else {
    string mmat_path = "/home/nnrthmr/CLionProjects/ma_thesis/data/demos/translationManip3d.csv";
    MatrixXd data(80, 10);
    MatrixXd dataVectorized(80, 7);
    load_data_mmat_skip_first(mmat_path, &data);
    deb(data);

    for (int i = 0; i < data.rows(); i++) {
        dataVectorized(i, 0) = data(i,0);
        MatrixXd tmp = data.block(i, 1, 1, 9);
        tmp.resize(3, 3);
        dataVectorized.block(i, 1, 1, 6) = Symmat2Vec(tmp);
    }

    deb(dataVectorized);


    GMM_SPD model2 = GMM_SPD();
    model2.InitModel(dataVectorized);
    model2.TrainEM();

    MatrixXd expData(6, 20);
    expData.setZero();
    std::vector<MatrixXd> expSigma; //m_dimOut x m_dimOut x m_nData

    std::cout<<"GMR"<<std::endl;
    model2.GMR(expData, expSigma);

    WriteCSV(Vec2Symmat(expData), "/home/nnrthmr/CLionProjects/ma_thesis/data/expDataSPD3d.csv");
}



//    WriteCSV(expSigma, "/home/nnrthmr/CLionProjects/ma_thesis/data/expSigma.csv");
//
//    Franka robot = Franka();
//    VectorXd q_goal(7);
//    q_goal << -pi/2.0, 0.004, 0.0, -1.57156, 0.0, 1.57075, 0.0;
//
////    robot.moveToQGoal(q_goal);
//MatrixXd J = robot.getTranslationJacobian();
//deb(J);


//MatrixXd jtfull(6,4);
//jtfull <<-2.00000000000000,	2.00000000000000,	5.46410161513775,	3.46410161513776,
//    3.46410161513776,	3.46410161513776,	1.46410161513776,	-2.00000000000000,
//    0,	0,	0,	0,
//    0,	0,	0,	0,
//    0,	0,	0,	0,
//    1,	1,	1,	1;
//
//MatrixXd med(2,2);
//med <<304.380134285343,	-188.288919971871,
//        -188.288919971871,	124.941294748085;
//
//MatrixXd mct(2,2);
//mct << 49.8564064605510,	1.07179676972449,
//        1.07179676972449,	30.1435935394490;
//
//
//vector<MatrixXd> t;
//t.push_back(med);
//MatrixXd mdiff = LogMap(med, mct);
//
//MatrixXd xd(2,1);
//xd <<0,0;
//
//MatrixXd x(2,1);
//xd <<0.3,0.3;
//
//robot.ManipulabilityTrackingSecondaryTask(x, xd, med);

return 0;
}
