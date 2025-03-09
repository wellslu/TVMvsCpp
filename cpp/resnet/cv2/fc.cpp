#include "modules.hpp"

FullyConnectedLayer::FullyConnectedLayer() {};

FullyConnectedLayer::FullyConnectedLayer(string name, int in_features, int out_features)
    : name(name), in_features(in_features), out_features(out_features) {
      };

void FullyConnectedLayer::load_weights(const cnpy::npz_t &npz_data)
{
    string key_name = name + ".weight";
    auto it = npz_data.find(key_name);
    if (it != npz_data.end())
    {
        this->weights = cv::Mat(out_features, in_features, CV_32F);
        memcpy(this->weights.data, (it->second).data<float>(), sizeof(float) * (this->weights.size().height) * (this->weights.size().width));
        cout << "successfully load " << "\"" << key_name << "\": \"" << this->weights.size() << "\"" << endl;
    }
    else
    {
        cout << "Failed to load " << key_name << endl;
    }

    key_name = name + ".bias";
    it = npz_data.find(key_name);
    if (it != npz_data.end())
    {
        this->bias = cv::Mat(out_features, 1, CV_32F);
        memcpy(this->bias.data, (it->second).data<float>(), sizeof(float) * (this->bias.size().height) * (this->bias.size().width));
        cout << "successfully load " << "\"" << key_name << "\": \"" << this->bias.size() << "\"" << endl;
    }
    else
    {
        cout << "Failed to load " << key_name << endl;
    }
}

cv::Mat FullyConnectedLayer::forward(const cv::Mat &input)
{
    // Check the input and weights dimensions for matrix multiplication
    CV_Assert(input.cols == weights.cols); // input size must match weight input size

    // Perform matrix multiplication: output = input * weights^T
    cv::Mat output = input * weights.t(); // Weights are transposed to match the dimensions

    // Add the bias to each row of the output (broadcast bias)
    cv::Mat bias_reshaped = bias.reshape(1, 1); // Reshape bias to [1, output_size]
    cv::Mat final_output = output + bias_reshaped;

    return final_output;
}