#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "part3.hpp"
#include "json.hpp"
using json = nlohmann::json;
using namespace std;
// 从二进制文件加载矩阵
template<typename T>
Matrix<T> loadMatrix(const string& filepath, int rows, int cols) {
    Matrix<T> mat(rows, cols);
    ifstream file(filepath, ios::binary);
    if (!file.is_open()) throw runtime_error("Failed to open ");
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            T value;
            file.read((char*)(&value), sizeof(T));//二进制读取文件
            if (!file) throw runtime_error("Failed to read data " + filepath);
            mat.data[i][j] = value;
        }
    }
    return mat;
}

// 加载模型
template<typename T>
Model<T> loadModel(const string& path) {
    ifstream metaFile(path + "/meta.json");
    if (!metaFile.is_open()) throw runtime_error("Cannot open meta.json");

    json meta;
    metaFile >> meta;

    int w1r = meta["fc1.weight"][0], w1c = meta["fc1.weight"][1];
    int b1r = meta["fc1.bias"][0],   b1c = meta["fc1.bias"][1];
    int w2r = meta["fc2.weight"][0], w2c = meta["fc2.weight"][1];
    int b2r = meta["fc2.bias"][0],   b2c = meta["fc2.bias"][1];

    Matrix<T> weight1 = loadMatrix<T>(path + "/fc1.weight", w1r, w1c);
    Matrix<T> bias1   = loadMatrix<T>(path + "/fc1.bias",   b1r, b1c);
    Matrix<T> weight2 = loadMatrix<T>(path + "/fc2.weight", w2r, w2c);
    Matrix<T> bias2   = loadMatrix<T>(path + "/fc2.bias",   b2r, b2c);

    return Model<T>(weight1, bias1, weight2, bias2);
}
// int main() {
//     try {
//         auto model = loadModel<float>("C:/Users/czx/Desktop/GKD-Software-2025-Test/mnist-fc-plus"); // 假设模型保存在 model 文件夹中

//         cout << "Weight1 X:"<< model.getWeight1().rows << endl;
       

//         cout << "Weight1 Y:"<< model.getBias1().cols << endl;

//     } catch (const exception& e) {
//         cerr << "Error: " << e.what() << endl;
//     }

//     return 0;
// }

