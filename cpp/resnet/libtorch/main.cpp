// #include <torch/script.h>
// #include <opencv2/opencv.hpp>
// #include <string>
// #include <iostream>

// using namespace std;
// using namespace cv;

// const string WEIGHTS_FOLDER = "../../../data/MNIST/";
// const string IMAGE_PATH = "../../../3.png";

// torch::Tensor preprocess_image(const string &image_path)
// {
//     // 1. 读取图片 (灰度模式)
//     Mat img = imread(image_path, IMREAD_GRAYSCALE);
//     if (img.empty())
//     {
//         cerr << "Error: Could not read image " << image_path << endl;
//         exit(-1);
//     }

//     // 2. 调整大小到 28x28
//     resize(img, img, Size(28, 28));

//     // 3. 转换为 float 并归一化到 [0, 1]
//     img.convertTo(img, CV_32F, 1.0 / 255.0);

//     // 4. 应用均值和标准差归一化
//     float mean = 0.1307f; // MNIST 数据集的均值
//     float std = 0.3081f;  // MNIST 数据集的标准差

//     // 对于灰度图像（单通道），直接进行归一化
//     img = (img - mean) / std;

//     // 5. 转换 OpenCV Mat 为 torch::Tensor
//     torch::Tensor tensor_image = torch::from_blob(img.data, {1, 1, 224, 224}, torch::kFloat32).clone();

//     // 返回归一化后的 tensor
//     return tensor_image;
// }

// int main()
// {
//     // 1. 加载 TorchScript 模型
//     std::string model_path = WEIGHTS_FOLDER + "resnet18_my_jit.pth";
//     torch::jit::script::Module model;

//     try
//     {
//         model = torch::jit::load(model_path);
//         model.to(torch::kCPU);
//         std::cout << "Model loaded successfully!\n";
//     }
//     catch (const c10::Error &e)
//     {
//         std::cerr << "Error loading the model: " << e.what() << std::endl;
//         return -1;
//     }

//     // 2. 创建输入 Tensor
//     // torch::Tensor input_tensor = preprocess_image(IMAGE_PATH);
//     // cout << input_tensor.sizes() << endl;

//     torch::Tensor input_tensor = torch::rand({1, 3, 224, 224}).to(torch::kCPU);
//     std::vector<torch::jit::IValue> inputs = {input_tensor};

//     // 3. 运行推理
//     // std::vector<torch::jit::IValue> inputs;
//     // inputs.push_back(input_tensor);

//     at::Tensor output = model.forward(inputs).toTensor();

//     // 4. 输出结果
//     std::cout << "Model output: " << output << std::endl;

//     return 0;
// }

#include <torch/script.h>
#include <iostream>

int main()
{
    // 加载模型
    torch::jit::script::Module module;
    try
    {
        module = torch::jit::load("../../../data/MNIST/resnet18_my_jit.pth");
        module.to(torch::kCPU); // 确保模型在 CPU 上
    }
    catch (const c10::Error &e)
    {
        std::cerr << "Error loading the model\n";
        return -1;
    }

    std::cout << "Model loaded successfully on CPU\n";

    // 创建输入张量并推理
    torch::Tensor input_tensor = torch::rand({1, 3, 224, 224}).to(torch::kCPU);
    std::vector<torch::jit::IValue> inputs = {input_tensor};
    at::Tensor output = module.forward(inputs).toTensor();

    std::cout << "Output: " << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
}