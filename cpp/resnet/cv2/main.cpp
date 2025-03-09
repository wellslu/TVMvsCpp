#include <cstring>
#include <cstdio>
#include <iostream>
#include "resnet.hpp"
using namespace std;

const string WEIGHTS_FOLDER = "../../../data/MNIST/";

cv::Mat get_input(string image_path)
{
    cv::Mat raw_image = cv::imread(image_path, cv::IMREAD_GRAYSCALE); // 可以指定读取方式，如彩色（IMREAD_COLOR）或灰度（IMREAD_GRAYSCALE）
    cv::Mat resizedImage;
    cv::resize(raw_image, resizedImage, cv::Size(28, 28)); // 目标尺寸是 28x28

    cv::Mat floatImage;
    resizedImage.convertTo(floatImage, CV_32F, 1.0 / 255.0);

    // Normalize: (x - mean) / std
    // 均值和标准差
    float mean = 0.1307f;
    float std = 0.3081f;

    // 将每个像素值减去均值并除以标准差
    cv::Mat input = (floatImage - mean) / std;
    return input;
}

void test_resnet34()
{
    std::string imagePath = "../../../3.png";
    cv::Mat input = get_input(imagePath);

    ResNet resnet34 = ResNet("ResNet-34", 34, 10);
    resnet34.load_weights(WEIGHTS_FOLDER + "resnet34.npz");

    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat logits = resnet34.forward(input);
    auto end = std::chrono::high_resolution_clock::now();

    int max_idx = 0;
    float max_logit = 0;
    for (int i = 0; i < logits.cols; ++i)
    {
        float logit = logits.ptr<float>(0, i)[0]; // 使用ptr来访问每个元素
        if (logit > max_logit)
        {
            max_logit = logit;
            max_idx = i;
        }
    }
    cout << logits.size() << endl;
    cout << logits << endl;
    cout << "ResNet 34 Prediction: " << max_idx << endl;

    std::chrono::duration<double> duration = end - start;
    std::cout << "Inference time: " << duration.count() << " seconds" << std::endl;
}

void test_resnet18()
{
    std::string imagePath = "../../../3.png";
    cv::Mat input = get_input(imagePath);

    ResNet resnet18 = ResNet("ResNet-18", 34, 10);
    resnet18.load_weights(WEIGHTS_FOLDER + "resnet18.npz");

    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat logits = resnet18.forward(input);
    auto end = std::chrono::high_resolution_clock::now();

    int max_idx = 0;
    float max_logit = 0;
    for (int i = 0; i < logits.cols; ++i)
    {
        float logit = logits.ptr<float>(0, i)[0]; // 使用ptr来访问每个元素
        if (logit > max_logit)
        {
            max_logit = logit;
            max_idx = i;
        }
    }
    cout << logits.size() << endl;
    cout << logits << endl;
    cout << "ResNet 34 Prediction: " << max_idx << endl;

    std::chrono::duration<double> duration = end - start;
    std::cout << "Inference time: " << duration.count() << " seconds" << std::endl;
}

int main()
{
    test_resnet34();
    return 0;
}