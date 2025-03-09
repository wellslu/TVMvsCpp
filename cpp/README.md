# C++ Model Code

## Target
### pure CPP implementation (WIP)  
  - Strengths:
    - less dependancy
    - compatible with more platforms
  - Weakness:
    - computation inefficiency
      - **malloc**: can't align the allocated the data automatically with 64 bytes (compatible with SIMD instructions -> better performance)
### TorchLib implementation (Pending)
### OpenBlas implementation (Pending)
### MKL (won't implement as only supported on Intel CPUs)


## Step
- [ ] 🎯 Develop different version of ResNet and MobileNet models using C++
  - [ ] ⏳ **(WIP)** ResNet-18/34/50
  - [ ] MobileNet(v1\~v3)
- [ ] 🛠️ Load weights of models trained using PyTorch
- [ ] 🏗️ Deploy models and weights to **raspberry pi**
- [ ] 📊 Compare the efficiency and accuracy of models with TVM
  