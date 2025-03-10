import torch
import torchvision.models as models


# 1. get a model
model = models.resnet18(pretrained=True).to("cpu")

# 2. use torch.jit.trace to convert the model
example_input = torch.rand(1, 3, 224, 224)  # needs an exmaple input
traced_model = torch.jit.trace(model, example_input)

# 3. save as TorchScript models
traced_model.save("model_jit.pth")  # ✅ can be loaded by lib_torch
