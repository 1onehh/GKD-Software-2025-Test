#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "part1.hpp"
#include "json.hpp"
using json = nlohmann::json;
using namespace std;
// 从二进制文件加载矩阵
Matrix loadMatrix(const string& filepath, int rows, int cols) {
    Matrix mat(rows, cols);
    ifstream file(filepath, ios::binary);
    if (!file.is_open()) throw runtime_error("Failed to open ");
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float value;
            file.read((char*)(&value), sizeof(float));//二进制读取文件
            if (!file) throw runtime_error("Failed to read data " + filepath);
            mat.data[i][j] = value;
        }
    }
    return mat;
}

// 加载模型

Model loadModel(const string& path) {
    ifstream metaFile(path + "/meta.json");
    if (!metaFile.is_open()){
        throw runtime_error("Cannot open meta.json");
    }
    json meta;
    metaFile >> meta;
    int w1r = meta["fc1.weight"][0];
    int w1c = meta["fc1.weight"][1];
    int b1r = meta["fc1.bias"][0];
    int b1c = meta["fc1.bias"][1];
    int w2r = meta["fc2.weight"][0];
    int w2c = meta["fc2.weight"][1];
    int b2r = meta["fc2.bias"][0];
    int b2c = meta["fc2.bias"][1];
    Matrix weight1 = loadMatrix(path + "/fc1.weight", w1r, w1c);
    Matrix bias1   = loadMatrix(path + "/fc1.bias",   b1r, b1c);
    Matrix weight2 = loadMatrix(path + "/fc2.weight", w2r, w2c);
    Matrix bias2   = loadMatrix(path + "/fc2.bias",   b2r, b2c);

    return Model(weight1, bias1, weight2, bias2);
};

// int main() {
//     try {
//         Model model = loadModel("C:/Users/czx/Desktop/GKD-Software-2025-Test/mnist-fc");
//         cout << "Model loaded successfully." <<endl;
//     } catch (const exception& e) {
//         cerr << "Load failed: " << e.what() << endl;
//     }
// }
