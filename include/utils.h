#ifndef MA_THESIS_UTILS_H
#define MA_THESIS_UTILS_H

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/QR>
#include <unsupported/Eigen/CXX11/Tensor>
#include <fstream>
#include <cstdlib>
#include <iostream>
#include <unsupported/Eigen/MatrixFunctions>
#include <sys/stat.h>

#define deb(x) cout << #x << " " << x << endl;
using namespace std;
using namespace Eigen;
const static IOFormat CSVFormat(StreamPrecision, DontAlignCols, ", ", "\n");

inline
bool fileExists (const std::string& name) {
    struct stat buffer;
    return (stat (name.c_str(), &buffer) == 0);
}

inline
int loadCSV (const string path, vector<vector<double>> &values) {
    ifstream indata;
//    ios_base::iostate exceptionMask = indata.exceptions() | std::ios::failbit;
//    indata.exceptions(exceptionMask);
//    try{
        indata.open(path);
        string line;
        uint rows = 0;
        while (getline(indata, line)) {
            vector<double> values_i;
            stringstream lineStream(line);
            string cell;
//            uint cols=0;
            while (getline(lineStream, cell, ',')) {
//            if(cols<2){
//                cols++;
//                continue;
//            }
                values_i.push_back(stod(cell));
            }
            values.push_back(values_i);
            ++rows;
        }
        indata.close();
//    }
//    catch (std::ios_base::failure& e) {
//        std::cerr << e.what() << '\n';
//    }

    return 0;
}

inline
int loadCSVSkipFirst (const string path, vector<vector<double>> values) {
    ifstream indata;
    indata.exceptions ( ifstream::failbit | ifstream::badbit );
    try{
        indata.open(path);
        string line;
        uint rows = 0;

        while (getline(indata, line)) {
            if(rows<1){
                ++rows;
                continue;
            }
            vector<double> values_i;
            stringstream lineStream(line);
            string cell;
            uint cols=0;
            while (getline(lineStream, cell, ',')) {
//            if(cols<2){
//                cols++;
//                continue;
//            }
                values_i.push_back(stod(cell));
            }
            values.push_back(values_i);
            ++rows;
        }
        indata.close();
    }
    catch (const ifstream::failure& e) {
        cout << "Exception opening/reading file";
        throw;
    }

    return 0;
}

inline
void writeCSV(const MatrixXd& data, const string path){
    ofstream outdata;
    outdata.open(path);
    outdata << data.format(CSVFormat);
    outdata.close();
}

/**
 * Writes data to csv file, rowwise. Matrices are rolled out into one row per matrix.
 * @param data
 * @param path
 */
inline
void writeCSV(const vector<MatrixXd>& data, const string path){
    ofstream outdata;
    MatrixXd tmp(data.size(), data[0].cols()*data[0].rows());
    tmp.setZero();

    MatrixXd tmp2(1, data[0].rows()*data[0].cols());


    for(int i=0;i<data.size();i++){
        for(int j=0; j< data[0].rows();j++){
            for(int k=0; k< data[0].cols();k++){
                tmp2(0,(j*data[i].cols()+k)) = data[i](j,k);
            }
        }
        tmp.row(i) = tmp2;
    }

    outdata.open(path);
    outdata << tmp.format(CSVFormat);
    outdata.close();
}



//TODO resample spline(), maybe preprocess data by using promp-ias/utils.py interpolation
inline
MatrixXd read_cartesian_trajectories(const string file_path){
    vector<vector<double>> data;
    loadCSV(file_path, data);
    MatrixXd trajs(4,data.size());
    trajs.setZero();
    for (int t = 0; t < data.size(); t++) {
        trajs(0, t) = t*0.01;
        trajs(1, t) = data[t][0];
        trajs(2, t) = data[t][1];
        trajs(3, t) = data[t][2];
    }
    return trajs;
}

/**
 *  First 12 values are the joint values
 *  values 13-15 are the wrist position
 *  values 16-18 are the wrist orientation in xyz
 *  values 19-54 are the wrist position jacobian (first row, then second row, then third row)
 *  values 55-90 are the wrist orientation jacobian (in the same format as above)
 */
inline
void readTxtFile(std::string fileName, MatrixXd *positions, MatrixXd *positionJacobians)
{
    MatrixXd data(8219, 90);
    data.setZero();
    std::ifstream in(fileName.c_str());
    if(!in)
    {
        std::cerr << "Cannot open the File : "<<fileName<<std::endl;
    }
    float item = 0.0;
    in >> item;
    in >> item; // first row ignore
    for (int row = 0; row < 8219; row++)
        for (int col = 0; col < 90; col++)
        {
            in >> item;
            data(row, col) = item;
        }
    (*positions) = data.block(0,12,8219,3);
    (*positionJacobians) = data.block(0,18,8219,36);
    in.close();
}



inline
void load_data_cmat(const string data_path, MatrixXd *data_pos){
    data_pos->setZero();

    vector<vector<double>> data;
    loadCSV(data_path, data);
    std::cout<<data.size()<<" "<<data[0].size()<<std::endl;
    for (int t = 0; t < 400; t++) {
        (*data_pos)(0, t) = data[0][t];
        (*data_pos)(1, t) = data[1][t];
        (*data_pos)(2, t) = data[2][t];
    }
}

inline
void loadCSV(const string data_path, MatrixXd *data_m){
    data_m->setZero();

    vector<vector<double>> data;
    loadCSV(data_path, data);
    for (int t = 0; t < data.size(); t++) {
        for (int c = 0; c < data[0].size(); c++) {
            (*data_m)(t, c) = data[t][c];
        }
    }
}

inline
void loadCSVSkipFirst(const string data_path, MatrixXd *data_m){
    data_m->setZero();

    vector<vector<double>> data;
    loadCSVSkipFirst(data_path, data);
    for (int t = 0; t < data.size(); t++) {
        for (int c = 0; c < data[0].size(); c++) {
            (*data_m)(t, c) = data[t][c];
        }
    }
}

// Checked!
inline
vector<int> linspace(double a, double b, size_t N) {
    double h = (b - a) / static_cast<double>(N - 1);
    vector<int> xs(N);
    vector<int>::iterator x;
    double val;
    for (x = xs.begin(), val = a; x != xs.end(); ++x, val += h) {
        *x = (int) round(val);
    }
    return xs;
}

// Checked!
inline
void cumulativeSum(const VectorXd &input, VectorXd &result) {
    result(0) = input[0];
    for (int i = 1; i < input.size(); i++) {
        result(i) = result(i - 1) + input(i);
    }
}

// Checked!
inline
MatrixXd parallelTransport(const MatrixXd& S1, const MatrixXd& S2) {
    MatrixXd S3;
    S3 = (S1.transpose().inverse() * S2.transpose()).transpose().pow(0.5); //B/A = (A'\B')' and A\B = A.inv()*B
    return S3;
}

// Checked!
inline
vector<MatrixXd> parallelTransport(const vector<MatrixXd>& S1, const vector<MatrixXd>& S2) {
    vector<MatrixXd> S3;
    for (int i = 0; i < S1.size(); i++) {
        S3.push_back(parallelTransport(S1[i], S2[i])); //B/A = (A'\B')' and A\B = A.inv()*B
    }
    return S3;
}

// Checked!
inline
MatrixXd symmat2Vec(const MatrixXd& mat) {
    int N = mat.rows();
    vector<double> v;
    VectorXd dia = mat.diagonal();
    for (int x = 0; x < dia.size(); x++) {
        v.push_back(dia(x));
    }
    int row, col;
    for (int n = 1; n < N; n++) {
        row = 0;
        col = n;
        for (int ni = n; ni < N; ni++) {
            v.push_back(sqrt(2) * mat(row, col));
            row++;
            col++;
        }
    }
    MatrixXd vm(1, v.size());
    for (int x = 0; x < v.size(); x++) {
        vm(0, x) = v[x]; //one row
    }
    return vm;
}

// Checked!
inline
vector<MatrixXd> symmat2Vec(const vector<MatrixXd>& mat_vec) {
    MatrixXd vn;
    int N = mat_vec.size();
    vector<MatrixXd> vec;
    for (int i = 0; i < N; i++) {
        vn = symmat2Vec(mat_vec[i]);
        vec.push_back(vn);
    }
    return vec;
}

// Checked!
inline
vector<MatrixXd> vec2Symmat(const MatrixXd& vec) {
    vector<MatrixXd> MVector;
    MatrixXd vn, Mn;
    int d = vec.rows();
    int N = vec.cols();
    int D = (-1 + sqrt(1 + 8 * d)) / (double) 2;
    VectorXd id(D);
    int row;
    for (int n = 0; n < N; n++) { //colwise
        vn = vec.col(n).transpose();
        Mn = vn.row(0).leftCols(D).asDiagonal();
        id.setZero();
        cumulativeSum(VectorXd::LinSpaced(D, D, 1), id);
        MatrixXd tmp1(Mn.rows(), Mn.cols());
        MatrixXd tmp2(Mn.rows(), Mn.cols());
        for (int i = 1; i < D; i++) {
            tmp1.setZero();
            row = 0;
            for (int k = i; k < id(i) - id(i - 1) + i; k++) {
                tmp1(row, k) = vn(0, id(i - 1) + row) * (1 / sqrt(2));
                row++;
            }
            tmp2.setZero();
            row = 0;
            for (int k = i; k < id(i) - id(i - 1) + i; k++) {
                tmp2(k, row) = vn(0, id(i - 1) + row) * (1 / sqrt(2));
                row++;
            }
            Mn = Mn + tmp1 + tmp2;
        }
        MVector.push_back(Mn);
    }
    return MVector;
}

// Checked!
inline
vector<MatrixXd> vec2Symmat(const vector<MatrixXd>& vec) {
    MatrixXd v(vec[0].rows(), vec.size());
    for (int i = 0; i < vec.size(); i++) {
        v.col(i) = vec[i];
    }
    return vec2Symmat(v);
}

// Checked!
inline
vector<MatrixXd> expMap(const vector<MatrixXd>& U, const MatrixXd& S) {
    vector<MatrixXd> X;
    MatrixXd D,V, tmp2;
    for (int i = 0; i < U.size(); i++) {
        MatrixXd tmp = (S.inverse()) * U[i]; //A\B in MATLAB is a^-1 * B
        EigenSolver<MatrixXd> es(tmp);
        D = es.eigenvalues().real().asDiagonal();
        V = es.eigenvectors().real();
        tmp2 = D.diagonal().array().exp().matrix().asDiagonal().toDenseMatrix();
        X.push_back(S * V * tmp2 * V.inverse());
    }
    return X;
}

// Checked!
inline
vector<MatrixXd> expmapVec(const MatrixXd& u, const MatrixXd& s) {
    vector<MatrixXd> U = vec2Symmat(u);
    vector<MatrixXd> S = vec2Symmat(s);
    vector<MatrixXd> X = expMap(U, S[0]); //vec2Symmat gives back vector of size 1 here
    vector<MatrixXd> x = symmat2Vec(X);
    return x;
}

// Checked!
inline
vector<MatrixXd> logMap(const vector<MatrixXd>& X, const MatrixXd& S) {
    vector<MatrixXd> U;
    MatrixXd tmp, D, V, tmp2;
    for (int i = 0; i < X.size(); i++) {
        tmp = (S.inverse()) * X[i]; //A\B in MATLAB is a^-1 * B
        EigenSolver<MatrixXd> es(tmp);
        D = es.eigenvalues().real().asDiagonal();
        V = es.eigenvectors().real();
        tmp2 = D.diagonal().array().log().matrix().asDiagonal().toDenseMatrix();
        U.push_back(S * V * tmp2 * V.inverse());
    }
    return U;
}

inline
MatrixXd logMap(const MatrixXd& X, const MatrixXd& S) {
    MatrixXd U;
    MatrixXd tmp = (S.inverse()) * X; //A\B in MATLAB is a^-1 * B
    EigenSolver<MatrixXd> es(tmp);
    MatrixXd D = es.eigenvalues().real().asDiagonal();
    MatrixXd V = es.eigenvectors().real();
    MatrixXd tmp2 = D.diagonal().array().log().matrix().asDiagonal().toDenseMatrix();
    U = S * V * tmp2 * V.inverse();
    return U;
}

// Checked!
inline
vector<MatrixXd> logMapVec(const MatrixXd& x, const MatrixXd& s) {
    vector<MatrixXd> X = vec2Symmat(x);
    vector<MatrixXd> S = vec2Symmat(s);
    vector<MatrixXd> U = logMap(X, S[0]); //vec2Symmat gives back vector of size 1 here
    vector<MatrixXd> u = symmat2Vec(U);
    return u;
}

// Checked!
inline
MatrixXd spdMean(const vector<MatrixXd>& mat, int nIter) {
    MatrixXd M = mat[0];
    MatrixXd tmp;
    MatrixXd L(mat[0].rows(), mat[0].cols());
    for (int iter = 0; iter < nIter; iter++) {
        L.setZero();
        for (int i = 0; i < mat.size(); i++) {
            tmp = M.pow(-0.5) * mat[i] * (M.pow(-0.5));
            L = L + (tmp.log()).matrix();
        }
        M = M.pow(0.5) * (L.array() / mat.size()).matrix().exp().matrix() * M.pow(0.5);
    }
    return M;
}

inline
MatrixXd getDiffVector(const vector<MatrixXd>& xHat, const MatrixXd& m, const int nPoints){
    int nDemos = m.rows()/nPoints;
    MatrixXd diffs(nDemos,nPoints);
    MatrixXd MDesired, M;
    for(int j=0;j<nDemos;j++){
        for(int i=0;i<nPoints;i++) {
            MDesired = m.row(j*nPoints+i).rightCols(9);
            MDesired.resize(3, 3);
            deb(MDesired)

            M = xHat[i];
            deb(M)
            diffs(j,i) = ((MDesired.pow(-0.5) * M * MDesired.pow(-0.5)).log().norm());
        }
    }
    return diffs;
}
#endif //MA_THESIS_UTILS_H
