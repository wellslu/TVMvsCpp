#include "modules.hpp"

ConvBN::ConvBN(string name, int in_channels, int out_channels, int kernel_size, int stride, int padding) : name(name), conv(name + ".0", in_channels, out_channels, kernel_size, stride, padding, false), bn(name + ".1", out_channels) {};

void ConvBN::load_weights(const cnpy::npz_t &npz_data)
{
    this->conv.load_weights(npz_data);
    this->bn.load_weights(npz_data);
}

vector<cv::Mat> ConvBN::forward(const vector<cv::Mat> &input)
{
    vector<cv::Mat> output = input;
    cout << "conv forwarding\n";
    output = this->conv.forward(output);
    cout << "bn forwarding\n";
    output = this->bn.forward(output);
    return output;
}
