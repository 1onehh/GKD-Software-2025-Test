#include <vector>
#include <thread>
#include <stdexcept>
#include <functional>
#include <chrono>
#include <iostream>
#include "part2moban.hpp"
#include "part3.hpp"

using namespace std;

// part4.cpp 中
template<typename T>
class ThreadedMatrix {
public:
    std::vector<std::vector<T>> data;
    int rows, cols;

    ThreadedMatrix(int r, int c) : rows(r), cols(c) {
        data = std::vector<std::vector<T>>(r, std::vector<T>(c, 0));
    }

    // 多线程矩阵乘法函数
    static ThreadedMatrix<T> multiply(const ThreadedMatrix<T>& A, const ThreadedMatrix<T>& B, int thread_count = std::thread::hardware_concurrency()) {
        if (A.cols != B.rows)
            throw std::invalid_argument("Matrix dimensions do not match for multiplication");

        ThreadedMatrix<T> result(A.rows, B.cols);

        auto multipalyRow = [&](int start_row, int end_row) {
            for (int i = start_row; i < end_row; ++i) {
                for (int j = 0; j < B.cols; ++j) {
                    T sum = 0;
                    for (int k = 0; k < A.cols; ++k) {
                        sum += A.data[i][k] * B.data[k][j];
                    }
                    result.data[i][j] = sum;
                }
            }
        };

        std::vector<std::thread> threads;
        int rows_per_thread = A.rows / thread_count;

        for (int i = 0; i < thread_count; ++i) {
            int start = i * rows_per_thread;
            int end = (i == thread_count - 1) ? A.rows : start + rows_per_thread;
            threads.emplace_back(multipalyRow, start, end);
        }

        for (auto& t : threads) t.join();

        return result;
    }
};


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
auto start = chrono::high_resolution_clock::now();
auto result = ThreadedMatrix<double>::multiply(A, B);
auto end = chrono::high_resolution_clock::now();

chrono::duration<double> duration = end - start;

auto start1 = chrono::high_resolution_clock::now();
auto result1 = Matrix<double>::multiply(A1, B1);
auto end1 = chrono::high_resolution_clock::now();

chrono::duration<double> duration1 = end1 - start1;
cout << "Multithreaded multiply time: " << duration.count() << " seconds" << endl;
cout << " multiply time: " << duration1.count() << " seconds" << endl;


}
