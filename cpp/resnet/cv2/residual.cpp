#include "modules.hpp"

// residual stage
ResidualStage::ResidualStage() {};

ResidualStage::ResidualStage(string name, int in_channels, int out_channels, int block_num, bool stage1, bool use_bottleneck)
{
    this->name = name;
    if (!use_bottleneck)
    {
        for (int i = 0; i < block_num; i++)
        {
            if (i == 0 && !stage1)
            {
                this->layers.push_back(ResidualBlock(name + "." + to_string(i), in_channels, out_channels, 2));
            }
            else
            {
                this->layers.push_back(ResidualBlock(name + "." + to_string(i), in_channels, out_channels));
            }
            in_channels = out_channels;
        }
    }
    else
    {
        for (int i = 0; i < block_num; i++)
        {
            if (i == 0 && !stage1)
            {
                this->bottleneck_layers.push_back(BottleneckBlock(name + "." + to_string(i), in_channels, out_channels, 2));
            }
            else
            {
                this->bottleneck_layers.push_back(BottleneckBlock(name + "." + to_string(i), in_channels, out_channels));
            }
            in_channels = out_channels;
        }
    }
}

void ResidualStage::load_weights(const cnpy::npz_t &npz_data)
{
    if (!(this->layers.empty()))
    {
        for (auto &layer : this->layers)
        {
            layer.load_weights(npz_data);
        }
    }
    if (!(this->bottleneck_layers.empty()))
    {
        for (auto &b_layer : this->bottleneck_layers)
        {
            b_layer.load_weights(npz_data);
        }
    }
    return;
}

vector<cv::Mat> ResidualStage::forward(const vector<cv::Mat> &input)
{
    vector<cv::Mat> output = input;
    if (!layers.empty())
    {
        for (auto &layer : layers)
        {
            output = layer.forward(output);
        }
    }

    if (!bottleneck_layers.empty())
    {
        int idx = 0;
        for (auto &b_layer : bottleneck_layers)
        {
            output = b_layer.forward(output);
            idx += 1;
        }
    }
    return output;
}

// residual block
ResidualBlock::ResidualBlock(string name, int in_channels, int out_channels, int stride)
    : name(name), cb1(name + ".CB1", in_channels, out_channels, 3, stride), cb2(name + ".CB2", out_channels, out_channels), shortcut(nullptr)
{
    if (in_channels != out_channels)
    {
        this->shortcut = new ConvBN(name + ".shortcut", in_channels, out_channels, 1, stride, 0);
    }
}

ResidualBlock::ResidualBlock(ResidualBlock &&other) noexcept : name(std::move(other.name)), cb1(std::move(other.cb1)),
                                                               cb2(std::move(other.cb2)),
                                                               shortcut(std::move(other.shortcut))
{
    other.shortcut = nullptr;
}

ResidualBlock::~ResidualBlock()
{
    if (this->shortcut)
    {
        delete this->shortcut;
    }
    return;
}

void ResidualBlock::load_weights(const cnpy::npz_t &npz_data)
{
    this->cb1.load_weights(npz_data);
    this->cb2.load_weights(npz_data);
    if (this->shortcut)
    {
        this->shortcut->load_weights(npz_data);
    }
};

vector<cv::Mat> ResidualBlock::forward(const vector<cv::Mat> &input)
{
    vector<cv::Mat> output;
    vector<cv::Mat> identity;
    for (const auto &mat : input)
    {
        output.push_back(mat.clone());   // 深拷贝
        identity.push_back(mat.clone()); // 深拷贝
    }

    ReLU relu;

    output = this->cb1.forward(output);
    output = relu.forward(output);
    output = this->cb2.forward(output);

    if (this->shortcut)
    {
        identity = this->shortcut->forward(identity);
    }

    for (size_t i = 0; i < output.size(); ++i)
    {
        output[i] += identity[i]; // 逐元素相加
    }

    output = relu.forward(output);

    return output;
}

BottleneckBlock::BottleneckBlock(string name, int in_channels, int out_channels, int stride) : name(name), cb1(name + ".CB1", in_channels, out_channels / 4, 1, 1, 0), cb2(name + ".CB2", out_channels / 4, out_channels / 4, 3, stride), cb3(name + ".CB3", out_channels / 4, out_channels, 1, 1, 0), shortcut(nullptr)
{
    if (stride != 1 || in_channels != out_channels)
    {
        this->shortcut = new ConvBN(name + ".shortcut", in_channels, out_channels, 1, stride, 0);
    }
}

BottleneckBlock::BottleneckBlock(BottleneckBlock &&other) noexcept : name(std::move(other.name)), cb1(std::move(other.cb1)),
                                                                     cb2(std::move(other.cb2)), cb3(std::move(other.cb3)),
                                                                     shortcut(std::move(other.shortcut))
{
    other.shortcut = nullptr;
}

BottleneckBlock::~BottleneckBlock()
{
    if (this->shortcut)
    {
        delete this->shortcut;
    }
    return;
}

void BottleneckBlock::load_weights(const cnpy::npz_t &npz_data)
{
    this->cb1.load_weights(npz_data);
    this->cb2.load_weights(npz_data);
    this->cb3.load_weights(npz_data);
    if (this->shortcut)
    {
        this->shortcut->load_weights(npz_data);
    }
    return;
}

vector<cv::Mat> BottleneckBlock::forward(const vector<cv::Mat> &input)
{
    ReLU relu;
    vector<cv::Mat> identity;
    for (const auto &mat : input)
    {
        identity.push_back(mat.clone()); // 深拷贝
    }
    vector<cv::Mat> output = this->cb1.forward(input);
    output = relu.forward(output);
    output = this->cb2.forward(output);
    output = relu.forward(output);
    output = this->cb3.forward(output);
    if (this->shortcut)
    {
        identity = this->shortcut->forward(identity);
    }
    for (size_t i = 0; i < output.size(); ++i)
    {
        output[i] += identity[i];
    }
    output = relu.forward(output);
    return output;
}
