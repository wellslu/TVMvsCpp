#include <cstring>
#include <cstdio>
#include <iostream>
#include "resnet.hpp"
using namespace std;

const string WEIGHTS_FOLDER = "../../../../tvm_model_ckpts/";

vector<cv::Mat> get_input(string image_path)
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
    cv::Mat one_channel_image = (floatImage - mean) / std;
    vector<cv::Mat> input;
    input.push_back(one_channel_image);
    return input;
}

void test_resnet(int arch)
{
    string arch_str = to_string(arch);
    int repeat_times = 5;

    std::string imagePath = "../../../3.png";
    vector<cv::Mat> input = get_input(imagePath);

    cout << "------------------------------------------------------------------------------\n";
    ResNet resnet = ResNet("ResNet-" + arch_str, arch, 10);
    resnet.load_weights(WEIGHTS_FOLDER + "ResNet" + arch_str + ".npz");

    for (int idx = 0; idx < repeat_times; idx++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        cv::Mat logits = resnet.forward(input);
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

        std::cout << "Repeat-" << idx << std::endl;
        cout << "\t" << logits << endl;
        cout << "\tResNet-" + arch_str + " Prediction: " << max_idx << endl;

        std::chrono::duration<double> duration = end - start;
        std::cout << "\tResNet-" + arch_str + " Inference time : " << duration.count() << " seconds \n"
                  << std::endl;
    }
    cout << "------------------------------------------------------------------------------\n";
    return;
}

int main()
{
    test_resnet(18);
    test_resnet(34);
    test_resnet(50);
    test_resnet(101);
    test_resnet(152);

    return 0;
}