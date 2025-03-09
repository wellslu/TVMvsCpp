#include "modules.hpp"

Conv2DLayer::Conv2DLayer(string name, int in_channels, int out_channels, int kernel_size, int stride, int padding, bool bias)
    : name(name), in_channels(in_channels), out_channels(out_channels), kernel_size(kernel_size), stride(stride), padding(padding), bias(bias) {};

void Conv2DLayer::load_weights(const cnpy::npz_t &npz_data)
{

    string key_name = name + ".weight";
    static auto it = npz_data.find(key_name);
    if (it != npz_data.end())
    {
        this->weights = cv::Mat(out_channels, in_channels * kernel_size * kernel_size, CV_32F);
        memcpy(this->weights.data, (it->second).data<float>(), sizeof(float) * (this->weights.size().height) * (this->weights.size().width));
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
    it = cnpy::npz_t::const_iterator();
}

cv::Mat Conv2DLayer::forward(const cv::Mat &input)
{
    int in_height = input.rows, in_width = input.cols;
    assert(in_channels == input.channels());

    int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;

    cv::Mat paddedInput;
    cv::copyMakeBorder(input, paddedInput, padding, padding, padding, padding, cv::BORDER_CONSTANT, 0);

    std::vector<cv::Mat> inputChannels;
    cv::split(paddedInput, inputChannels);

    std::vector<cv::Mat> outputChannels(out_channels);
    for (int o = 0; o < out_channels; ++o)
        outputChannels[o] = cv::Mat(out_height, out_width, CV_32F, cv::Scalar(0));

    cv::Mat &kernels = this->weights;
    for (int o = 0; o < out_channels; ++o)
    {
        for (int i = 0; i < in_channels; ++i)
        {
            float *kernel_ptr = kernels.ptr<float>(o) + i * kernel_size * kernel_size;
            cv::Mat kernel(kernel_size, kernel_size, CV_32F, kernel_ptr);

            cv::Mat filtered;
            cv::filter2D(inputChannels[i], filtered, -1, kernel, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);

            // **手动进行 stride 采样**
            for (int h = 0; h < out_height; ++h)
            {
                for (int w = 0; w < out_width; ++w)
                {
                    outputChannels[o].at<float>(h, w) = filtered.at<float>(h * stride, w * stride);
                }
            }
        }

        if (this->bias)
            outputChannels[o] += biases.at<float>(o, 0);
    }

    cv::Mat output;
    cout << outputChannels.size() << endl;
    cv::merge(outputChannels, output);
    return output;
}