#include <vector>
#include <thread>
#include <stdexcept>
#include <chrono>
#include <iostream>
#include "part2moban.hpp"
#include "part3.hpp"

using namespace std;

// part4.cpp 中
template<typename T>
class ThreadedMatrix {
public:
    vector<vector<T>> data;
    int rows, cols;

    ThreadedMatrix(int r, int c) : rows(r), cols(c) {
        data = vector<vector<T>>(r, vector<T>(c, 0));
    }

    // 多线程矩阵乘法函数
    static ThreadedMatrix<T> multiply(const ThreadedMatrix<T>& A, const ThreadedMatrix<T>& B, int thread_count = thread::hardware_concurrency()) {
        if (A.cols != B.rows)
            throw invalid_argument("dimensions do not match");

        ThreadedMatrix<T> result(A.rows, B.cols);
        auto multiplyRow = [&](int startrow, int endrow) {
            for (int i = startrow; i < endrow; ++i) {
                for (int j = 0; j < B.cols; ++j) {
                    T sum = 0;
                    for (int k = 0; k < A.cols; ++k) {
                        sum += A.data[i][k] * B.data[k][j];
                    }
                    result.data[i][j] = sum;
                }
            }
        };

        vector<thread> threads;
        int rows_per_thread = A.rows / thread_count;

         for (int i = 0; i < thread_count; ++i) {
            int start = i * rows_per_thread;
            int end = (i == thread_count - 1) ? A.rows : start + rows_per_thread;
            threads.push_back(thread(multiplyRow, start, end));
        }

        for (thread & t : threads)
            t.join();

        // cout << "Launched threads: " << threads.size() << endl;
        // cout << "Hardware concurrency: " << thread::hardware_concurrency() << endl;
        return result;
    }
};
template<typename T>
class ModelBaseThread{
public:
     virtual vector<T> forwardThread(const ThreadedMatrix<T>& input) = 0;
};

template<typename T>
class ModelThread :public ModelBaseThread<T>{
private:
    ThreadedMatrix<T> weight1, bias1, weight2, bias2;//784 * 500  1 * 500  500 * 10  1 * 10
public:
    ModelThread(const ThreadedMatrix<T>& w1, const ThreadedMatrix<T>& b1, const ThreadedMatrix<T>& w2, const ThreadedMatrix<T>& b2): weight1(w1), bias1(b1), weight2(w2), bias2(b2) {}
    vector<T> forwardThread(const ThreadedMatrix<T>& input) override{
        Matrix<T> z1 = ThreadedMatrix<T>::add(ThreadedMatrix<T>::multiply(input, weight1), bias1);
        Matrix<T> a1 = relu(z1);
        Matrix<T> z2 = ThreadedMatrix<T>::add(ThreadedMatrix<T>::multiply(a1, weight2), bias2);
        return softmax(z2.data[0]);
    }
};
//这里我不知道测试的对不对我只对多线程优化矩阵乘法进行测试记录时间，没有对forward进行测试，因为我感觉这样需要实现model和一些操作的多线程版本
//，过于复杂，可能也是我想的不对
//test
int main() {
    ThreadedMatrix<double> A(784, 500);
    ThreadedMatrix<double> B(500, 1);
    Matrix<double> A1(784, 500);
    Matrix<double> B1(500, 1);
   
    for (int i = 0; i < 784; ++i)
        for (int j = 0; j < 500; ++j)
            A.data[i][j] = i + j;

    for (int i = 0; i < 500; ++i)
        for (int j = 0; j < 1; ++j)
            B.data[i][j] = i - j;

            for (int i = 0; i < 784; ++i)
        for (int j = 0; j < 500; ++j)
            A1.data[i][j] = i + j;

    for (int i = 0; i < 500; ++i)
        for (int j = 0; j < 1; ++j)
            B1.data[i][j] = i - j;
    Model<double>* model = new Model<double>(loadModel<double>("C:/Users/czx/Desktop/GKD-Software-2025-Test/mnist-fc-plus"));
    
// 构造模拟输入
        Matrix<double> input(1, 784);
        for (int i = 0; i < 784; ++i) {
            input.data[0][i] = (double)(i % 256) / 255.0;
        }

        // 测试用时
        auto start2 = std::chrono::high_resolution_clock::now();
        vector<double> output = model->forward(input);
        auto end2 = std::chrono::high_resolution_clock::now();
        chrono::duration<double> duration2 = end2 - start2;

        cout << "multiply time: " << duration2.count() << "s" << endl;

auto start = chrono::high_resolution_clock::now();
auto result = ThreadedMatrix<double>::multiply(A, B);
auto end = chrono::high_resolution_clock::now();
chrono::duration<double> duration = end - start;

auto start1 = chrono::high_resolution_clock::now();
auto result1 = Matrix<double>::multiply(A1, B1);
auto end1 = chrono::high_resolution_clock::now();
chrono::duration<double> duration1 = end1 - start1;

cout << "Multithreaded multiply time: " << duration.count() << "s" << endl;
cout << " multiply time: " << duration1.count() << "s" << endl;



//多线程优化后时间：0.00192s 不优化：0.004813s 仅仅对于矩阵乘法来说

}
