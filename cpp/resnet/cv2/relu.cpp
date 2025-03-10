#include "modules.hpp"

vector<cv::Mat> ReLU::forward(const vector<cv::Mat> &input)
{
    vector<cv::Mat> output;
    for (const auto &mat : input)
    {
        cv::Mat relu_mat = mat.clone();
        cv::max(relu_mat, 0, relu_mat); // 逐个矩阵应用 ReLU
        output.push_back(relu_mat);
    }
    return output;
}
