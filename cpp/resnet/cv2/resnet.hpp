#ifndef MODELS_HPP
#define MODELS_HPP

#include <iostream>
#include <vector>
#include <array>
#include <cnpy.h>
#include <cmath>
#include <string>
#include <unordered_map>
#include "modules.hpp"

const std::unordered_map<int, std::vector<std::vector<int>>> archs = {
    {18, {{64, 64, 2}, {64, 128, 2}, {128, 256, 2}, {256, 512, 2}}},
    {34, {{64, 64, 3}, {64, 128, 4}, {128, 256, 6}, {256, 512, 3}}},
    {50, {{64, 256, 3}, {256, 512, 4}, {512, 1024, 6}, {1024, 2048, 3}}},
    {101, {{64, 256, 3}, {256, 512, 4}, {512, 1024, 23}, {1024, 2048, 3}}},
    {152, {{64, 256, 3}, {256, 512, 8}, {512, 1024, 36}, {1024, 2048, 3}}}};

// ResNet34定义
class ResNet
{
public:
    ResNet(string name, int arch, int num_classes);
    void load_weights(const std::string &weight_file);
    cv::Mat forward(const vector<cv::Mat> &input);

private:
    string name;
    int arch;
    int num_classes;

    // self.conv1 in torch
    Conv2DLayer conv1;
    BatchNormLayer bn1;
    ReLU relu;
    MaxPool2DLayer max_pool1;

    // self.stage 1-4 in torch
    ResidualStage stage1, stage2, stage3, stage4;

    // self.pool in torch
    AdaptiveAvgPool2DLayer adap_pool;

    FullyConnectedLayer fc;
    SoftmaxLayer softmax;
};

#endif