#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <algorithm>
#include "part2moban.cpp"

using namespace cv;
using namespace std;

const int CANVAS_SIZE = 280;  // 10倍28x28，方便手绘
const int RESIZE_SIZE = 28;

Mat canvas(CANVAS_SIZE, CANVAS_SIZE, CV_8UC1, Scalar(255));  // 白底
mutex canvas_mutex;

atomic<bool> running(true);

Model<double> model = loadModel<double>("C:/Users/czx/Desktop/GKD-Software-2025-Test/mnist-fc-plus");

vector<double> current_probs(10, 0.0);
int current_pred = -1;

bool drawing = false;
Point last_point;

// 鼠标事件处理，左键画黑色线条
void on_mouse(int event, int x, int y, int, void*) {
    if (event == EVENT_LBUTTONDOWN) {
        drawing = true;
        last_point = Point(x, y);
    } else if (event == EVENT_MOUSEMOVE && drawing) {
        Point pt(x, y);
        lock_guard<mutex> lock(canvas_mutex);
        line(canvas, last_point, pt, Scalar(0), 15, LINE_AA);  // 画黑线，线宽15
        last_point = pt;
    } else if (event == EVENT_LBUTTONUP) {
        drawing = false;
    }
}

// 后台线程：定时处理画布，做模型预测
void predict_loop() {
    while (running) {
        Mat img;
        {
            lock_guard<mutex> lock(canvas_mutex);
            canvas.copyTo(img);
        }

        // 处理图像：转成28x28，白底黑字
        Mat gray;
        resize(img, gray, Size(RESIZE_SIZE, RESIZE_SIZE));

        // 白底黑字（当前是黑字白底，不用反色）  
        // 但是题目说要白底黑字且数字是0~1的浮点数
        // 先转换成浮点，归一化0~1，数字为黑（0），背景白（1），需要反色：
        Mat float_img;
        gray.convertTo(float_img, CV_64F, 1.0/255.0);
        // 反色，使数字为白（1），背景为0，符合你要求
        float_img = 1.0 - float_img;

        // 构造模型输入 vector<vector<double>>
        vector<vector<double>> data(1, vector<double>(RESIZE_SIZE*RESIZE_SIZE));
        for (int i = 0; i < RESIZE_SIZE; ++i) {
            for (int j = 0; j < RESIZE_SIZE; ++j) {
                data[0][i * RESIZE_SIZE + j] = float_img.at<double>(i, j);
            }
        }

        Matrix<double> input(data);
        vector<double> output = model.forward(input);

        if (!output.empty()) {
            current_probs = output;
            current_pred = max_element(output.begin(), output.end()) - output.begin();
        }

        this_thread::sleep_for(chrono::milliseconds(200));  // 每200ms预测一次
    }
}

// 绘制概率柱状图
void draw_prob_bar(Mat& display) {
    int bar_w = 20;
    int bar_h_max = 150;
    int x0 = CANVAS_SIZE + 30;
    int y0 = 30;

    for (int i = 0; i < 10; ++i) {
        int bar_h = static_cast<int>(current_probs[i] * bar_h_max);
        rectangle(display,
                  Point(x0 + i * (bar_w + 10), y0 + bar_h_max - bar_h),
                  Point(x0 + i * (bar_w + 10) + bar_w, y0 + bar_h_max),
                  Scalar(0, 0, 255), FILLED);
        putText(display, to_string(i), Point(x0 + i * (bar_w + 10), y0 + bar_h_max + 20),
                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 0), 2);
    }
}

int main() {
    cout << "Loading model..." << endl;
    model = loadModel<double>("C:/Users/czx/Desktop/GKD-Software-2025-Test/mnist-fc-plus");
    cout << "Model loaded." << endl;

    namedWindow("Draw Digits", WINDOW_AUTOSIZE);
    setMouseCallback("Draw Digits", on_mouse);

    // 启动预测线程
    thread pred_thread(predict_loop);

    while (true) {
        Mat display(CANVAS_SIZE, CANVAS_SIZE + 300, CV_8UC3, Scalar(255,255,255));
        {
            lock_guard<mutex> lock(canvas_mutex);
            cvtColor(canvas, display(Rect(0, 0, CANVAS_SIZE, CANVAS_SIZE)), COLOR_GRAY2BGR);
        }

        // 绘制预测结果
        if (current_pred >= 0) {
            string pred_text = "Predicted: " + to_string(current_pred);
            putText(display, pred_text, Point(CANVAS_SIZE + 20, 220),
                    FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0,0,0), 2);
            draw_prob_bar(display);
        }

        imshow("Draw Digits", display);

        char key = (char)waitKey(30);
        if (key == 27) { // ESC键退出
            break;
        } else if (key == 'c' || key == 'C') { // 按C清空画布
            lock_guard<mutex> lock(canvas_mutex);
            canvas.setTo(Scalar(255));
            current_pred = -1;
            fill(current_probs.begin(), current_probs.end(), 0);
        }
    }

    running = false;
    pred_thread.join();

    return 0;
}
