#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>

using namespace std;

class Matrix {
public:
    vector<vector<float>> data;
    int rows, cols;
    Matrix(int r, int c) : rows(r), cols(c) {
        data = vector<vector<float>>(r, vector<float>(c, 0.0f));
    }
    Matrix(const vector<vector<float>>& d) {
        data = d;
        rows = d.size();
        cols = d.empty() ? 0 : d[0].size();
    }

    static Matrix multiply(const Matrix& A, const Matrix& B) {
        if (A.cols != B.rows)
            throw invalid_argument("dimensions do not match");
        Matrix result(A.rows, B.cols);
        for (int i = 0; i < A.rows; ++i) {
            for (int j = 0; j < B.cols; ++j) {
                for (int k = 0; k < A.cols; ++k) {
                    result.data[i][j] += A.data[i][k] * B.data[k][j];
                }
            }
        }
        return result;
    }

    static Matrix add(const Matrix& A, const Matrix& B) {
        if (A.rows != B.rows || A.cols != B.cols)
            throw invalid_argument("dimensions do not match");
        Matrix result(A.rows, A.cols);
        for (int i = 0; i < A.rows; ++i) {
            for (int j = 0; j < A.cols; ++j) {
                result.data[i][j] = A.data[i][j] + B.data[i][j];
            }
        }
        return result;
    }
};

// ReLU 
Matrix relu(const Matrix& input) {
    Matrix result = input;
    for (int i = 0; i < result.rows; ++i) {
        for (int j = 0; j < result.cols; ++j) {
            result.data[i][j] = max(0.0f, result.data[i][j]);
        }
    }
    return result;
};

// SoftMax 
vector<float> softmax(const vector<float>& input) {
    vector<float> output(input.size());
    float sum = 0.0f;
    for (float val : input) {
        sum += exp(val); 
    }
    for (int i = 0; i < input.size(); ++i) {
        output[i] = exp(input[i]) / sum;
    }
    return output;
};

class Model{
private:
    Matrix weight1, bias1, weight2, bias2;//784 * 500 1 * 500 500 * 10 1 * 10
public:
    Model(const Matrix& w1, const Matrix& b1, const Matrix& w2, const Matrix& b2): weight1(w1), bias1(b1), weight2(w2), bias2(b2) {}
    vector<float> forward(const Matrix& input) {
        Matrix z1 = Matrix::add(Matrix::multiply(input, weight1), bias1);
        Matrix a1 = relu(z1);
        Matrix z2 = Matrix::add(Matrix::multiply(a1, weight2), bias2);
        return softmax(z2.data[0]);
    }
};
