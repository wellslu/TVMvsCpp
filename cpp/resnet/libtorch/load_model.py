import torch


# 加载完整模型
resnet18 = torch.load("../../../data/MNIST/ResNet18_jit.pth")

# 将模型设置为评估模式
resnet18.eval()

# 示例输入
input_tensor = torch.rand(1, 1, 28, 28)

# 前向传播
with torch.no_grad():
    output = resnet18(input_tensor)

print(output)
