#ifndef MODULES_HPP
#define MODULES_HPP

#define debug(x) cerr << #x << "=" << x << endl
#include <opencv2/opencv.hpp>
#include <cnpy.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <cassert>
#include <cmath>
using namespace std;

class Conv2DLayer
{
public:
    Conv2DLayer(string name, int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 0, bool bias = true);
    void load_weights(const cnpy::npz_t &npz_data);
    cv::Mat forward(const cv::Mat &input);

private:
    string name;
    int in_channels, out_channels, kernel_size, stride, padding;
    bool bias;
    cv::Mat weights, biases;
};

class BatchNormLayer
{
public:
    BatchNormLayer(string name, int channels);
    void load_weights(const cnpy::npz_t &npz_data);
    cv::Mat forward(const cv::Mat &input);

private:
    string name;
    int channels;
    cv::Mat gamma, beta;
    cv::Mat running_mean, running_var;
};

class ConvBN
{
public:
    ConvBN(string name, int in_channels, int out_channels, int kernel_size = 3, int stride = 1, int padding = 1);
    void load_weights(const cnpy::npz_t &npz_data);

    cv::Mat forward(const cv::Mat &input);

private:
    string name;
    Conv2DLayer conv;
    BatchNormLayer bn;
};

class ReLU
{
public:
    cv::Mat forward(const cv::Mat &input);
};

class MaxPool2DLayer
{
public:
    MaxPool2DLayer(int pool_size, int stride, int padding = 0);

    cv::Mat forward(const cv::Mat &input);

private:
    int pool_size; // 池化窗口大小
    int stride;    // 步幅
    int padding;   // 填充
};

class AdaptiveAvgPool2DLayer
{
public:
    AdaptiveAvgPool2DLayer();
    AdaptiveAvgPool2DLayer(int output_height, int output_width);

    cv::Mat forward(const cv::Mat &input);

private:
    int output_height; // 输出图像的高度
    int output_width;  // 输出图像的宽度
};

class ResidualBlock
{
public:
    ResidualBlock(string name, int in_channels, int out_channels, int stride = 1);
    void load_weights(const cnpy::npz_t &npz_data);
    cv::Mat forward(const cv::Mat &input);
    ResidualBlock(ResidualBlock &&other) noexcept;
    ~ResidualBlock();

private:
    string name;
    ConvBN cb1, cb2;
    ConvBN *shortcut;
};

class BottleneckBlock
{
public:
    BottleneckBlock();
    BottleneckBlock(string name, int in_channels, int out_channels, int stride = 1);
    BottleneckBlock(BottleneckBlock &&other) noexcept;
    void load_weights(const cnpy::npz_t &npz_data);
    cv::Mat forward(const cv::Mat &input);
    ~BottleneckBlock();

private:
    string name;
    ConvBN cb1, cb2, cb3;
    ConvBN *shortcut;
};

class ResidualStage
{
public:
    ResidualStage();
    ResidualStage(string name, int in_channels, int out_channels, int block_num, bool stage1 = false, bool use_bottleneck = false);
    void load_weights(const cnpy::npz_t &npz_data);
    cv::Mat forward(const cv::Mat &input);

private:
    string name;
    vector<ResidualBlock> layers;
    vector<BottleneckBlock> bottleneck_layers;
};

class FullyConnectedLayer
{
public:
    string name;
    int in_features, out_features;
    cv::Mat weights;
    cv::Mat bias;

    // 构造函数，初始化权重和偏置
    FullyConnectedLayer();
    FullyConnectedLayer(string name, int in_features, int out_features);
    void load_weights(const cnpy::npz_t &npz_data);

    // 前向传播计算
    cv::Mat forward(const cv::Mat &input);
};

class SoftmaxLayer
{
public:
    cv::Mat forward(const cv::Mat &input);
};
#endif
