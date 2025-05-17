#include <iostream>
#include <vector>
#include <thread>
#include <cstring>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <unistd.h>
#include "part3.hpp"  // 假设你的 Model 和 Matrix 类在这里
#include "part2moban.hpp" // loadModel 定义在这里
//服务端接口
using namespace std;

const int PORT = 12345;

Matrix<double> receiveMatrix(int client_socket) {
    int rows, cols;
    read(client_socket, &rows, sizeof(int));
    read(client_socket, &cols, sizeof(int));

    Matrix<double> res(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double val;
            read(client_socket, &val, sizeof(double));
            res.data[i][j] = val;
        }
    }
    return res;
}

void sendVector(int client_socket, const vector<double>& vec) {
    int size = vec.size();
    write(client_socket, &size, sizeof(int));
    for (double v : vec) {
        write(client_socket, &v, sizeof(double));
    }
}

int main() {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);

    sockaddr_in address;
    
    bind(server_fd, (struct sockaddr*)&address, sizeof(address));
    listen(server_fd, 1);

    cout << "Server listening on port " << PORT << "..." << endl;

    int addrlen = sizeof(address);
    int client_socket = accept(server_fd, (struct sockaddr*)&address, (socklen_t*)&addrlen);
    cout << "Client connected." << endl;

    Model<double> model = loadModel<double>("C:/Users/czx/Desktop/GKD-Software-2025-Test/mnist-fc-plus");

    while (true) {
        Matrix<double> input = receiveMatrix(client_socket);
        vector<double> output = model.forward(input);
        sendVector(client_socket, output);
    }

    close(client_socket);
    close(server_fd);
    return 0;
}
