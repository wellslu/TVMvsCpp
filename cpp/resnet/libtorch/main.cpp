#include <torch/script.h>
#include <string>
#include <iostream>

using namespace std;

const string WEIGHTS_FOLDER = "/mnt/";

int main()
{
    // 1. 加载 TorchScript 模型
    string arch="152";
    std::string model_path = WEIGHTS_FOLDER + "ResNet"+arch+"_jit.pth";
    torch::jit::script::Module model;

    try
    {
        model = torch::jit::load(model_path);
        model.to(torch::kCPU);
        std::cout << "Model loaded successfully!\n";
    }
    catch (const c10::Error &e)
    {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
        return -1;
    }

    // 2. 创建输入 Tensor
    // torch::Tensor input_tensor = preprocess_image(IMAGE_PATH);
    torch::Tensor input_tensor = torch::rand({1, 1, 28, 28}).to(torch::kCPU);

    // cout << input_tensor.sizes() << endl;

    // torch::Tensor input_tensor = torch::rand({1, 3, 224, 224}).to(torch::kCPU);
    std::vector<torch::jit::IValue> inputs = {input_tensor};

    // 3. 运行推理
    // std::vector<torch::jit::IValue> inputs;
    // inputs.push_back(input_tensor);

    int repeat_times=10000;
    float total_time = 0;
    for(int i=0;i<repeat_times;i++)
    {
        torch::Tensor input_tensor = torch::rand({1, 1, 28, 28}).to(torch::kCPU);
        auto start = std::chrono::high_resolution_clock::now();
        at::Tensor output = model.forward(inputs).toTensor();
        auto end = std::chrono::high_resolution_clock::now();
    
        // 4. 输出结果
        // std::cout << "Model output: " << output << std::endl;
        std::chrono::duration<double> duration = end - start;
        // std::cout << "\tResNet-"  + arch+" Inference time : " << duration.count() << " seconds \n"
        //           << std::endl;
        total_time+=duration.count();
    }
    cout<<"ResNet- "<<arch<<" avg time: "<<total_time/repeat_times<<endl;

    return 0;
}
