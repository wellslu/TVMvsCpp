import json
import torch
import numpy as np


model_name = "ResNet152"
pth_file_path = f"../data//MNIST/{model_name}.pth"
npz_file_path = f"../data/MNIST/{model_name}.npz"
# 1. 加载保存的 .pth 文件（模型权重）
model = torch.load(
    pth_file_path, map_location=torch.device("cpu")
)  # 假设该文件保存的是整个模型
# 如果你只保存了 state_dict，加载方式如下：
# model = YourModelClass()  # 重新定义模型架构
# model.load_state_dict(torch.load('model.pth'))

# 2. 获取权重（如果是 state_dict，提取相应层的权重）
if isinstance(model, torch.nn.Module):
    state_dict = model.state_dict()  # 获取模型权重
    print("loaded data is model")
else:
    state_dict = model  # 如果是保存的 state_dict
    print("loaded data is model weights")

# 3. 将权重保存为 .npz 文件
# 使用 dictionary 形式将所有权重保存到一个压缩文件
npz_dict = {}
strcut_dict = {}
# for name, tensor in state_dict["model"].items():
for name, tensor in state_dict.items():
    if isinstance(tensor, torch.Tensor):
        npz_dict[name] = tensor.detach().cpu().numpy()
        strcut_dict[name] = f"{tensor.shape}"
    else:
        print(f"Skipping {name}: not a tensor ({type(tensor)})")
        if name == "model":
            print(tensor)
# 保存为 npz 文件
np.savez(npz_file_path, **npz_dict)


import json

with open(f"{model_name}.json", "w") as fp:
    json.dump(strcut_dict, fp, indent=4)

print("Model weights saved as .npz file.")
