#include <cstdio>
#include <numeric>
#include "modules.hpp"

BatchNormLayer::BatchNormLayer(string name, int channels) : name(name), channels(channels)
{
}

void BatchNormLayer::load_weights(const cnpy::npz_t &npz_data)
{
    string key_name = name + ".weight";
    static typename cnpy::npz_t::const_iterator it = npz_data.end(); // 初始化一次
    it = npz_data.find(key_name);
    if (it != npz_data.end())
    {
        this->gamma = cv::Mat(channels, 1, CV_32F);
        memcpy(this->gamma.data, (it->second).data<float>(), sizeof(float) * (this->gamma.size().height) * (this->gamma.size().width));
        // cout << "successfully load " << "\"" << key_name << "\": \"" << this->gamma.size() << "\"" << endl;
    }
    else
    {
        cout << "Failed to load " << key_name << endl;
    }

    key_name = name + ".bias";
    it = npz_data.find(key_name);
    if (it != npz_data.end())
    {
        this->beta = cv::Mat(channels, 1, CV_32F);
        memcpy(this->beta.data, (it->second).data<float>(), sizeof(float) * (this->beta.size().height) * (this->beta.size().width));
        // cout << "successfully load " << "\"" << key_name << "\": \"" << this->beta.size() << "\"" << endl;
    }
    else
    {
        cout << "Failed to load " << key_name << endl;
    }

    key_name = name + ".running_mean";
    it = npz_data.find(key_name);
    if (it != npz_data.end())
    {
        this->running_mean = cv::Mat(channels, 1, CV_32F);
        memcpy(this->running_mean.data, (it->second).data<float>(), sizeof(float) * (this->running_mean.size().height) * (this->running_mean.size().width));
        // cout << "successfully load " << "\"" << key_name << "\": \"" << this->running_mean.size() << "\"" << endl;
    }
    else
    {
        cout << "Failed to load " << key_name << endl;
    }

    key_name = name + ".running_var";
    it = npz_data.find(key_name);
    if (it != npz_data.end())
    {
        this->running_var = cv::Mat(channels, 1, CV_32F);
        memcpy(this->running_var.data, (it->second).data<float>(), sizeof(float) * (this->running_var.size().height) * (this->running_var.size().width));
        // cout << "successfully load " << "\"" << key_name << "\": \"" << this->running_var.size() << "\"" << endl;
    }
    else
    {
        cout << "Failed to load " << key_name << endl;
    }
    it = cnpy::npz_t::const_iterator();
}

vector<cv::Mat> BatchNormLayer::forward(const vector<cv::Mat> &input)
{
    // 获取输入尺寸
    const float epsilon = 1e-5;
    // int height = input[0].rows;
    // int width = input[0].cols;
    int channels = input.size(); // 获取通道数

    std::vector<cv::Mat> outputChannels(channels);

    // 逐通道归一化
    for (int c = 0; c < channels; ++c)
    {
        cv::Mat x = input[c];

        // BN 归一化
        cv::Mat x_hat = (x - running_mean.at<float>(c)) / std::sqrt(running_var.at<float>(c) + epsilon);

        // 乘 gamma，加 beta
        outputChannels[c] = x_hat * gamma.at<float>(c) + beta.at<float>(c);
    }

    return outputChannels;
}
