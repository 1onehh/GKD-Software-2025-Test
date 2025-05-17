#include <iostream>
#include <vector>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <unistd.h>
//发送端接口
using namespace std;

void sendMatrix(int sock, const vector<vector<double>>& mat) {
    int rows = mat.size();
    int cols = mat[0].size();
    write(sock, &rows, sizeof(int));
    write(sock, &cols, sizeof(int));
    for (const auto& row : mat) {
        for (double val : row) {
            write(sock, &val, sizeof(double));
        }
    }
}

vector<double> receiveVector(int sock) {
    int size;
    read(sock, &size, sizeof(int));
    vector<double> vec(size);
    for (int i = 0; i < size; ++i) {
        read(sock, &vec[i], sizeof(double));
    }
    return vec;
}

int main() {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in serv_addr;

    connect(sock, (struct sockaddr*)&serv_addr, sizeof(serv_addr));
    cout << "Connected to server.\n";

    vector<vector<double>> testMatrix(1, vector<double>(784, 0.5));
    sendMatrix(sock, testMatrix);

    vector<double> result = receiveVector(sock);
    cout << "Received result: ";
    for (double v : result) cout << v << " ";
    cout << endl;

    close(sock);
    return 0;
}
