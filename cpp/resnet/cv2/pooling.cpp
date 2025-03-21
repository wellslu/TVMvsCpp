#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "modules.hpp"

MaxPool2DLayer::MaxPool2DLayer(int pool_size, int stride, int padding)
    : pool_size(pool_size), stride(stride), padding(padding) {};

vector<cv::Mat> MaxPool2DLayer::forward(const vector<cv::Mat> &input)
{
    // input: (H, W, C)
    int padded_rows = input[0].rows + 2 * this->padding;
    int padded_cols = input[0].cols + 2 * this->padding;

    std::vector<cv::Mat> output_channels;

    for (auto &channel : input)
    {
        // 进行 padding
        cv::Mat padded_input;
        cv::copyMakeBorder(channel, padded_input, this->padding, this->padding, this->padding, this->padding, cv::BORDER_CONSTANT, 0);

        int out_rows = (padded_rows - this->pool_size) / this->stride + 1;
        int out_cols = (padded_cols - this->pool_size) / this->stride + 1;
        cv::Mat output(out_rows, out_cols, CV_32F, cv::Scalar(0)); // 结果矩阵

        for (int i = 0; i < out_rows; ++i)
        {
            for (int j = 0; j < out_cols; ++j)
            {
                cv::Rect roi(j * this->stride, i * this->stride, this->pool_size, this->pool_size);
                cv::Mat region = padded_input(roi);
                double min_val, max_val;
                cv::minMaxLoc(region, &min_val, &max_val);
                output.at<float>(i, j) = static_cast<float>(max_val);
            }
        }
        output_channels.push_back(output);
    }

    return output_channels;
}

AdaptiveAvgPool2DLayer::AdaptiveAvgPool2DLayer() {};

AdaptiveAvgPool2DLayer::AdaptiveAvgPool2DLayer(int output_height, int output_width)
    : output_height(output_height), output_width(output_width) {}

vector<cv::Mat> AdaptiveAvgPool2DLayer::forward(const vector<cv::Mat> &input)
{
    // input: (H, W, C)
    int in_height = input[0].rows;
    int in_width = input[0].cols;

    int kernel_height = in_height / this->output_height;
    int kernel_width = in_width / this->output_width;

    std::vector<cv::Mat> output_channels;

    for (auto &channel : input)
    {
        cv::Mat output(this->output_height, this->output_width, CV_32F, cv::Scalar(0));

        // 每个通道单独做 Adaptive Average Pooling
        for (int i = 0; i < this->output_height; ++i)
        {
            for (int j = 0; j < this->output_width; ++j)
            {
                int startX = j * kernel_width;
                int startY = i * kernel_height;
                int endX = std::min(startX + kernel_width, in_width);
                int endY = std::min(startY + kernel_height, in_height);

                // 计算池化区域的均值
                cv::Rect roi(startX, startY, endX - startX, endY - startY);
                cv::Mat region = channel(roi);

                double avg = cv::mean(region)[0];
                output.at<float>(i, j) = static_cast<float>(avg);
            }
        }
        output_channels.push_back(output);
    }

    return output_channels;
}
