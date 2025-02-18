# TVMvsCpp

## Problem
As AI models become more powerful, they continue to grow in size, efficiency often takes a backseat. However, large models aren’t always practical to deploy on edge devices—like those found in automatic vehicles, medical equipment such as ECG machines, and IoT sensors—due to limited processing power and memory. Ensuring fast inference speeds under these constraints is crucial. In this study, we explore two approaches—using TVM versus native C++—to implement AI models on edge devices, comparing their performance in terms of speed, accuracy, and memory usage from small to larger model. Our goal is to shed light on the trade-offs between model size and real-time requirements, providing insights that help strike the right balance in resource-constrained environments.

## Plan
- Train ResNet(18, 34, 50), MobileNet(v1\~v3), InceptionNet(v1\~v4) classifier by MNIST and CIFAR-10. (Probably just train some of these models, not all of them)
- Implement these AI models on raspberry pi 3 B+ by python code and TVM, check how much TVM can accelerate and could all of these models implement on device? 
- Extract model weights and develop in C++ source code.
- Compile C++ source code on raspberry pi 2 B+, and make sure the output is still the same as our expectation. 
- Compare two methods’(TVM and C++) inference speed, accuracy and memory cost. 

# TVM
tvm folder clone from https://github.com/apache/tvm
<br>
here is the install steps: 
```bash
git clone --recursive https://github.com/apache/tvm tvm
cd tvm
mkdir build
cp cmake/config.cmake build/
cd build
cmake .. -DUSE_LLVM=ON
make -j$(nproc)
cd ../python
pip install -e .
```
!!! libstdc++.so.6 version should be upgraded, include GLIBCXX_3.4.30. 
<br>
!!! check libstdc++.so.6, libtvm.so path

## Step
- [ ] TVM compile file
- [ ] TVM inference file
- [ ] Dockerfile

# raspberry pi 3 B+
- 4 cores
- RAM 900 Mb
- Swp 200 Mb
- storage 32 Gb