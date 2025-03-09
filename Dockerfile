# Use Ubuntu 20.04 (Stable & Better ARM Cross Compilation Support)
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install essential dependencies (including wget & CMake)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl python3 python3-dev python3-setuptools python3-pip \
    llvm-9 llvm-9-dev clang-9 \
    make g++ \
    libc6-dev-arm64-cross \
    qemu-user qemu-user-static \
    libtinfo-dev zlib1g-dev libedit-dev libxml2-dev \
    gcc-aarch64-linux-gnu g++-aarch64-linux-gnu \
    binutils-aarch64-linux-gnu \
    git vim cmake openssh-client \
    && rm -rf /var/lib/apt/lists/* 


# Upgrade CMake to version 3.22.2 (if default is outdated)
RUN arch=$(uname -m) && \
    if [ "$arch" = "x86_64" ]; then \
      echo "Detected x86_64 architecture ($arch)"; \
      wget https://github.com/Kitware/CMake/releases/download/v3.22.2/cmake-3.22.2-linux-x86_64.tar.gz && \
      tar -xzf cmake-3.22.2-linux-x86_64.tar.gz -C /usr/local --strip-components=1 && \
      rm -rf cmake-3.22.2-linux-x86_64.tar.gz; \
    else \
      echo "Detected ARM architecture ($arch)"; \
      wget https://github.com/Kitware/CMake/releases/download/v3.22.2/cmake-3.22.2-linux-aarch64.tar.gz && \
      tar -xzf cmake-3.22.2-linux-aarch64.tar.gz -C /usr/local --strip-components=1 && \
      rm -rf cmake-3.22.2-linux-aarch64.tar.gz; \
    fi

# Verify CMake version
RUN cmake --version

# Clone TVM
# ARG TVM_VERSION=main
# ARG GIT_URL=https://github.com/apache/tvm
# RUN git clone --recursive ${GIT_URL} /tvm
# WORKDIR /tvm
# RUN git checkout ${TVM_VERSION}
COPY tvm /tvm

# Prepare build directory
RUN rm -rf /tvm/build && mkdir -p /tvm/build
WORKDIR /tvm/build

# # Cross-compile TVM for Raspberry Pi 3B (ARM Cortex-A53)
RUN cmake .. \
    -DUSE_LLVM=OFF \
    -DUSE_CPP_RPC=ON \
    -DUSE_LIBBACKTRACE=OFF \
    -DTVM_BUILD_MAIN=OFF \
    -DTVM_BUILD_RUNTIME=ON \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
    -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc \
    -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ \
    -DCMAKE_C_FLAGS="-march=armv8-a -mtune=cortex-a53" \
    -DCMAKE_CXX_FLAGS="-march=armv8-a -mtune=cortex-a53" \
    -DCMAKE_BUILD_TYPE=Release \
    || (cat CMakeFiles/CMakeError.log && false)

# Run make to build TVM runtime
RUN make runtime -j$(nproc) || (cat CMakeFiles/CMakeError.log && false)

# Set environment variables
ENV TVM_HOME=/tvm
ENV PYTHONPATH=PYTHONPATH=$TVM_HOME/python
ENV LD_LIBRARY_PATH=$TVM_HOME/build:$LD_LIBRARY_PATH


RUN pip --version
RUN pip3 --version

# Install Python dependencies
RUN pip3 --no-cache-dir install \
        cython \ 
        numpy==1.24.4 \
        torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
        opencv-python==4.6.0.66 \
        pyyaml==6.0.2 \
        tqdm==4.64.0 \
        scipy==1.5.4 \
        requests==2.32.3 \
        mlconfig==0.1.7 mlflow==1.28.0 \
        Pillow==8.4.0 \
        entrypoints==0.4