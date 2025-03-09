import torch
import torchvision.models as models

# 加载预训练的 ResNet-18 模型
# resnet18 = models.resnet18(pretrained=True).to("cpu")
resnet18 = models.resnet18(pretrained=True).to("cpu")

# 将模型转换为 TorchScript 格式
scripted_model = torch.jit.script(resnet18)

# 保存模型
scripted_model.save("../../../data/MNIST/resnet18_my_jit.pth")
