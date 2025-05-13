#pragma once
#include <vector>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

using namespace std;

template <typename T>
class Matrix{
public:
    vector<vector<T>> data;
    int rows, cols;
    Matrix(int r, int c) : rows(r), cols(c) {
        data = vector<vector<T>>(r, vector<T>(c, (T)0.0));
    }
    Matrix(const vector<vector<T>>& d) {
        data = d;
        rows = d.size();
        cols = d.empty() ? 0 : d[0].size();
    }

    static Matrix<T> multiply(const Matrix<T>& A, const Matrix<T>& B) {
        if (A.cols != B.rows)
            throw invalid_argument("dimensions do not match");
        Matrix<T> result(A.rows, B.cols);
        for (int i = 0; i < A.rows; ++i) {
            for (int j = 0; j < B.cols; ++j) {
                for (int k = 0; k < A.cols; ++k) {
                    result.data[i][j] += A.data[i][k] * B.data[k][j];
                }
            }
        }
        return result;
    }

    static Matrix<T> add(const Matrix<T>& A, const Matrix<T>& B) {
        if (A.rows != B.rows || A.cols != B.cols)
            throw invalid_argument("dimensions do not match");
        Matrix<T> result(A.rows, A.cols);
        for (int i = 0; i < A.rows; ++i) {
            for (int j = 0; j < A.cols; ++j) {
                result.data[i][j] = A.data[i][j] + B.data[i][j];
            }
        }
        return result;
    }

};

//因为在model函数里面会调用relu和softmax,所以把这两个函数也模板化
//ReLu 模板
template<typename T>
Matrix<T> relu(const Matrix<T>& input) {
    Matrix<T> result = input;
    for (int i = 0; i < result.rows; ++i) {
        for (int j = 0; j < result.cols; ++j) {
            result.data[i][j] = max(T(0), result.data[i][j]);
        }
    }
    return result;
};
// SoftMax 模板
template<typename T>
vector<T> softmax(const vector<T>& input) {
    vector<T> output(input.size());
    float sum = 0.0f;
    for (float val : input) {
        sum += exp(val); 
        }
    for (int i = 0; i < input.size(); ++i) {
        output[i] = exp(input[i]) / sum;
        }
    return output;
    }

template<typename T>
class ModelBase{
     virtual vector<T> forward(const Matrix<T>& input) = 0;
};

template<typename T>
class Model :public ModelBase<T>{
private:
    Matrix<T> weight1, bias1, weight2, bias2;//784 * 500  1 * 500  500 * 10  1 * 10
public:
    Model(const Matrix<T>& w1, const Matrix<T>& b1, const Matrix<T>& w2, const Matrix<T>& b2): weight1(w1), bias1(b1), weight2(w2), bias2(b2) {}
    vector<T> forward(const Matrix<T>& input) override{
        Matrix<T> z1 = Matrix<T>::add(Matrix<T>::multiply(input, weight1), bias1);
        Matrix<T> a1 = relu(z1);
        Matrix<T> z2 = Matrix<T>::add(Matrix<T>::multiply(a1, weight2), bias2);
        return softmax(z2.data[0]);
    }
};
