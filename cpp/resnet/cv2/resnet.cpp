#include <iostream>
#include <vector>
#include <array>
#include <cnpy.h>
#include <cmath>
#include <string>
#include "modules.hpp"
#include "resnet.hpp"
using namespace std;

cv::Mat flattenChannelsToLongVector(const cv::Mat &input)
{
    // input: (H, W, C) multi-channel matrix

    // 拆分多通道矩阵
    std::vector<cv::Mat> channels;
    cv::split(input, channels); // 将多通道矩阵拆分成单通道矩阵

    // 用来存储拼接后的长向量
    std::vector<cv::Mat> flattenedChannels;

    // 展平每个通道并加入到 flattenedChannels 中
    for (size_t i = 0; i < channels.size(); ++i)
    {
        cv::Mat flat;
        channels[i].reshape(1, 1).copyTo(flat); // 展平通道，变为一个长向量
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

    int fc_in_size = layer_size_list[3][1];
    if (!(arch == 18 || arch == 34))
    {
        fc_in_size *= 4;
    }
    this->fc = FullyConnectedLayer("classifier.0", fc_in_size, num_classes);
}

cv::Mat ResNet::forward(const cv::Mat &input)
{

    cv::Mat output;
    cout << "start conv1.forward\n ";
    output = this->conv1.forward(input);
    cout << "start bn1.forward\n ";
    output = this->bn1.forward(output);
    cout << "start relu.forward\n ";
    output = this->relu.forward(output);
    cout << "start max_pool1.forward\n ";
    output = this->max_pool1.forward(output);
    cout << "start stage1.forward\n ";
    output = this->stage1.forward(output);
    cout << "start stage2.forward\n ";
    output = this->stage2.forward(output);
    cout << "start stage3.forward\n ";
    output = this->stage3.forward(output);
    cout << "start stage4.forward\n ";
    output = this->stage4.forward(output);
    cout << "start adap_pool.forward\n ";
    output = this->adap_pool.forward(output);

    output = flattenChannelsToLongVector(output);
    cout << "start fc.forward\n ";
    output = this->fc.forward(output);
    output = this->softmax.forward(output);
    return output;
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
