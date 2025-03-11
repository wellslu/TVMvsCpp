# C++ Implementation

We explore two different approaches for implementing ResNet models loading and inference in C++:

- Pure C++ Implementation with Some Performance Optimizations.
  - Strengths: 
    - **1)** Fewer dependencies.
    - **2)** More platform compatibility.
  - Weakness:  
    - **1)** Development is more challenging due to the lack of convenient data structures (e.g., tensors) and operations (e.g., convolution, pooling). 
    - **2)** Computational inefficiency, despite C++ being a relatively fast programming language itself. Achieving optimal performance requires substantial fine-grained optimization.
- LibTorch implementation.
  - Strengths: 
    - **1)** Easier development, with a rich set of data structures and functions.
    - **2)** High efficiency, benefiting from extensive underlying optimizations.
  - Weakness:  
    - **1)** Setting up the compilation environment can be tricky on specific machines (e.g., on a Mac with an M1 chip).
