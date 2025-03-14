#include <iostream>
#include <vector>
#include <array>
#include <cnpy.h>
#include <cmath>
#include <string>
#include "modules.hpp"
#include "resnet.hpp"
using namespace std;

cv::Mat flattenChannelsToLongVector(const vector<cv::Mat> &input)
{
    // input: (H, W, C) multi-channel matrix

    // 用来存储拼接后的长向量
    std::vector<cv::Mat> flattenedChannels;

    // 展平每个通道并加入到 flattenedChannels 中
    for (size_t i = 0; i < input.size(); ++i)
    {
        cv::Mat flat;
        input[i].reshape(1, 1).copyTo(flat); // 展平通道，变为一个长向量
        flattenedChannels.push_back(flat);
    }

    // 拼接所有的长向量成一个大向量
    cv::Mat longVector;
    cv::hconcat(flattenedChannels, longVector); // 横向拼接所有的长向量

    return longVector; // 返回拼接后的长向量
}

ResNet::ResNet(string name, int arch, int num_classes)
    : name(name), arch(arch), num_classes(num_classes), conv1("conv1.0", 1, 64, 7, 2, 3, false), bn1("conv1.1", 64), relu(), max_pool1(3, 2, 1)
{
    assert(arch == 18 || arch == 34 || arch == 50 || arch == 101 || arch == 152);
    std::cout << "ResNet-" << arch << " architecture:\n";

    // 确保 archs 里包含 key
    auto it = archs.find(arch);
    if (it == archs.end())
    {
        throw std::runtime_error("Unsupported ResNet architecture: " + std::to_string(arch));
    }
    auto layer_size_list = it->second;

    bool use_bottleneck = (arch == 50 || arch == 101 || arch == 152);

    this->stage1 = ResidualStage("stage1", layer_size_list[0][0], layer_size_list[0][1], layer_size_list[0][2], true, use_bottleneck);
    this->stage2 = ResidualStage("stage2", layer_size_list[1][0], layer_size_list[1][1], layer_size_list[1][2], false, use_bottleneck);
    this->stage3 = ResidualStage("stage3", layer_size_list[2][0], layer_size_list[2][1], layer_size_list[2][2], false, use_bottleneck);
    this->stage4 = ResidualStage("stage4", layer_size_list[3][0], layer_size_list[3][1], layer_size_list[3][2], false, use_bottleneck);

    this->adap_pool = AdaptiveAvgPool2DLayer(1, 1);

    this->fc = FullyConnectedLayer("classifier.0", layer_size_list[3][1], num_classes);
}

cv::Mat ResNet::forward(const vector<cv::Mat> &input)
{

    vector<cv::Mat> output;
    output = this->conv1.forward(input);
    output = this->bn1.forward(output);
    output = this->relu.forward(output);
    output = this->max_pool1.forward(output);
    output = this->stage1.forward(output);
    output = this->stage2.forward(output);
    output = this->stage3.forward(output);
    output = this->stage4.forward(output);
    output = this->adap_pool.forward(output);

    auto flatten_output = flattenChannelsToLongVector(output);
    flatten_output = this->fc.forward(flatten_output);
    flatten_output = this->softmax.forward(flatten_output);
    return flatten_output;
}

void ResNet::load_weights(const std::string &weight_file)
{
    cnpy::npz_t npz_data = cnpy::npz_load(weight_file);
    // cnpy::npz_t *npz_data = new cnpy::npz_t(cnpy::npz_load(weight_file));

    // size_t total_size = 0;
    // for (const auto &pair : npz_data)
    // {
    //     size_t array_size = pair.second.num_bytes(); // 获取该数组的字节数
    //     total_size += array_size;
    //     std::cout << "Key: " << pair.first << ", Size: " << array_size / (1024.0 * 1024.0) << " MB" << std::endl;
    // }
    // std::cout << "Total npz_data size: " << total_size / (1024.0 * 1024.0) << " MB" << std::endl;
    this->conv1.load_weights(npz_data);
    this->bn1.load_weights(npz_data);
    this->stage1.load_weights(npz_data);
    this->stage2.load_weights(npz_data);
    this->stage3.load_weights(npz_data);
    this->stage4.load_weights(npz_data);
    this->fc.load_weights(npz_data);
    return;
}
