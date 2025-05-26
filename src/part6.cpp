#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include "part2moban.cpp"


using namespace std;
using namespace cv;

int main() {
    cout << "Start loading model..." << endl;
    Model<double> model = loadModel<double>("C:/Users/czx/Desktop/GKD-Software-2025-Test/mnist-fc-plus");
    cout << "Model loaded." << endl;

    for (int i = 0; i < 10; ++i) {
        string filename = "C:/Users/czx/Desktop/GKD-Software-2025-Test/nums/" + to_string(i) + ".png";
        Mat img = imread(filename, IMREAD_GRAYSCALE);
        if (img.empty()) {
        cerr << "Failed to load image: " << filename << endl;
        continue;
        } else {
        cout << "Loaded image: " << filename << endl;
             }


        resize(img, img, Size(28, 28));
        img.convertTo(img, CV_64F); // 转为 double 类型

        // 拍扁成 1x784 矩阵并归一化
        vector<vector<double>> data(1, vector<double>(784));
        for (int i = 0; i < 28; ++i) {
            for (int j = 0; j < 28; ++j) {
                data[0][i * 28 + j] = img.at<double>(i, j) / 255.0;
            }
        }

        Matrix<double> input(data);
        vector<double> output = model.forward(input);

        // 输出结果
        int pred = max_element(output.begin(), output.end()) - output.begin();
        cout << i << ", Predicted: " << pred << ", Prob: ";
        for (double v : output) cout << v << " ";
        cout << endl;

        if (pred != i) {
            cout << "Warning: Prediction mismatch for " << i << ". Check model or input!" << endl;
        }
    }


    return 0;
}