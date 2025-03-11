# TVMvsCpp

## Problem
As AI models become more powerful, they continue to grow in size, efficiency often takes a backseat. However, large models aren’t always practical to deploy on edge devices—like those found in automatic vehicles, medical equipment such as ECG machines, and IoT sensors—due to limited processing power and memory. Ensuring fast inference speeds under these constraints is crucial. In this study, we explore two approaches—using TVM versus native C++—to implement AI models on edge devices, comparing their performance in terms of speed, accuracy, and memory usage from small to larger model. Our goal is to shed light on the trade-offs between model size and real-time requirements, providing insights that help strike the right balance in resource-constrained environments.

## Plan
- Train ResNet(18, 34, 50, 101, 152) classifier by MNIST. 
- Cross-compile TVM models to raspberry pi 3 B+.
- Extract model weights and develop in C++ source code.
- Compile C++ source code on raspberry pi 3 B+. 
- Compare 3 methods’(Pytorch, TVM and C++) inference speed and memory cost. 

# Git Clone
clone all moudule by: 
<br>
```bash
git clone --recursive https://github.com/wellslu/TVMvsCpp.git
```

# TVM (compile on x86)
here is the steps to install compile and runtime components: 
<br>
```bash
# git clone --recursive https://github.com/apache/tvm tvm
cd tvm
mkdir build
cp cmake/config.cmake build/
cd build
cmake .. -DUSE_LLVM=ON
make -j$(nproc)
cd ../python
pip install -e .
```

# raspberry pi 3 B+
- 4 cores
- aarch64 (not 32-bit armv7!!!)
- RAM 900 Mb
- storage 32 Gb

# Docker (Cross-Compilation)
You can build it on x86 first, and move libtvm_runtime.so to the raspberry pi 3 b. It also can be built on the raspberry pi 3 b directly if your device won't crash. 
<br>
```bash
docker build -t tvm_image .
docker run -it --name tvm_crossCompile tvm_image
```