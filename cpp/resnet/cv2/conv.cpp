#include "modules.hpp"

Conv2DLayer::Conv2DLayer(string name, int in_channels, int out_channels, int kernel_size, int stride, int padding, bool bias)
    : name(name), in_channels(in_channels), out_channels(out_channels), kernel_size(kernel_size), stride(stride), padding(padding), bias(bias) {};

void Conv2DLayer::load_weights(const cnpy::npz_t &npz_data)
{

    string key_name = name + ".weight";
    auto it = npz_data.find(key_name);
    if (it != npz_data.end())
    {
        this->weights = cv::Mat(out_channels, in_channels * kernel_size * kernel_size, CV_32F);
        memcpy(this->weights.data, (it->second).data<float>(), sizeof(float) * (this->weights.size().height) * (this->weights.size().width));
        cout << out_channels << " " << in_channels << " " << kernel_size << " ";
        cout << "successfully load " << "\"" << key_name << "\": \"" << this->weights.size() << "\"" << endl;
    }
    else
    {
        cout << "Failed to load " << key_name << endl;
    }

    if (this->bias)
    {
        string key_name = name + ".bias";
        auto it = npz_data.find(key_name);
        if (it != npz_data.end())
        {
            this->biases = cv::Mat(out_channels, 1, CV_32F);
            memcpy(this->biases.data, (it->second).data<float>(), sizeof(float) * (this->biases.size().height) * (this->biases.size().width));
            cout << "successfully load " << "\"" << key_name << "\": \"" << this->biases.size() << "\"" << endl;
        }
        else
        {
            cout << "Failed to load " << key_name << endl;
        }
    }
}

cv::Mat Conv2DLayer::forward(const cv::Mat &input)
{
    // input (H, W, C)
    int in_height = input.rows;
    int in_width = input.cols;
    assert(in_channels == input.channels());

    // **计算输出尺寸（考虑 Padding 和 Stride）**
    int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;

    // **初始化输出张量**
    cv::Mat output(out_height, out_width, CV_32FC(out_channels), cv::Scalar(0));

    // **输入 Padding 处理**
    cv::Mat paddedInput;
    cv::copyMakeBorder(input, paddedInput, padding, padding, padding, padding, cv::BORDER_CONSTANT, 0);

    // 将输入分为不同的通道
    std::vector<cv::Mat> inputChannels;
    cv::split(paddedInput, inputChannels);

    cv::Mat &kernels = this->weights; // kernels (out_channels, in_channels * kernel_size * kernel_size)

    // **执行卷积**
    for (int o = 0; o < out_channels; ++o)
    {
        cv::Mat outFeatureMap(out_height, out_width, CV_32F, cv::Scalar(0)); // 当前输出通道的特征图

        for (int i = 0; i < in_channels; ++i)
        {
            // **提取卷积核**
            cv::Mat kernel(kernel_size, kernel_size, CV_32F, (void *)kernels.ptr<float>(o, i * kernel_size * kernel_size));

            // **对单个输入通道应用卷积，考虑Stride**
            for (int h = 0; h < out_height; ++h)
            {
                for (int w = 0; w < out_width; ++w)
                {
                    // 计算卷积窗口的起始位置
                    int startY = h * stride;
                    int startX = w * stride;

                    // 提取区域
                    cv::Rect roi(startX, startY, kernel_size, kernel_size);
                    cv::Mat region = inputChannels[i](roi);

                    // 进行卷积（内积）
                    double sum = cv::sum(region.mul(kernel))[0];
                    outFeatureMap.at<float>(h, w) += static_cast<float>(sum);
                }
            }
        }

        // **添加 Bias**
        if (this->bias)
        {
            outFeatureMap += biases.at<float>(o, 0);
        }

        // **将当前通道的结果存入输出矩阵**
        std::vector<cv::Mat> outputChannels;
        cv::split(output, outputChannels);
        outputChannels[o] = outFeatureMap; // 将当前输出特征图放入对应的输出通道
        cv::merge(outputChannels, output); // 合并所有输出通道
    }

    return output;
}