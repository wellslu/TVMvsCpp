#include "modules.hpp"

// residual stage
ResidualStage::ResidualStage() {};

ResidualStage::ResidualStage(string name, int in_channels, int out_channels, int block_num, bool stage1)
{
    this->name = name;
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

void ResidualStage::load_weights(const cnpy::npz_t &npz_data)
{
    for (auto &layer : this->layers)
    {
        layer.load_weights(npz_data);
    }
    return;
}

cv::Mat ResidualStage::forward(const cv::Mat &input)
{
    cv::Mat output = input;
    for (auto &layer : layers)
    {
        output = layer.forward(output);
    }
    return output;
}

// residual block
ResidualBlock::ResidualBlock(string name, int in_channels, int out_channels, int stride)
    : name(name), cb1(name + ".CB1", in_channels, out_channels, 3, stride), cb2(name + ".CB2", out_channels, out_channels), shortcut(NULL)
{
    if (in_channels != out_channels)
    {
        this->shortcut = new ConvBN(name + ".shortcut", in_channels, out_channels, 1, stride, 0);
    }
}

void ResidualBlock::load_weights(const cnpy::npz_t &npz_data)
{
    this->cb1.load_weights(npz_data);
    this->cb2.load_weights(npz_data);
    if (this->shortcut != NULL)
    {
        this->shortcut->load_weights(npz_data);
    }
};

cv::Mat ResidualBlock::forward(const cv::Mat &input)
{
    cv::Mat output = input;
    cv::Mat identity = input;

    ReLU relu;

    output = this->cb1.forward(output);
    output = relu.forward(output);
    output = this->cb2.forward(output);

    if (this->shortcut != NULL)
    {
        identity = this->shortcut->forward(identity);
    }

    output += identity;
    output = relu.forward(output);

    return output;
}