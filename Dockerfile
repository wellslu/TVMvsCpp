# Use Ubuntu 20.04 (Stable & Better ARM Cross Compilation Support)
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install essential dependencies (including wget & CMake)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl python3 python3-dev python3-setuptools python3-pip \
    llvm-9 llvm-9-dev clang-9 \
    make g++ \
    libc6-dev-armhf-cross \
    qemu-user qemu-user-static \
    libtinfo-dev zlib1g-dev libedit-dev libxml2-dev \
    gcc-arm-linux-gnueabihf g++-arm-linux-gnueabihf \
    binutils-arm-linux-gnueabihf \
    git vim cmake && rm -rf /var/lib/apt/lists/*

# Upgrade CMake to version 3.22.2 (if default is outdated)
RUN cd /tmp \
    && wget https://github.com/Kitware/CMake/releases/download/v3.22.2/cmake-3.22.2-linux-x86_64.tar.gz \
    && tar -xzf cmake-3.22.2-linux-x86_64.tar.gz -C /usr/local --strip-components=1 \
    && rm -rf cmake-3.22.2-linux-x86_64.tar.gz

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

# # Cross-compile TVM for Raspberry Pi 3B (ARMv7)
RUN cmake .. \
    -DUSE_LLVM=OFF \
    -DUSE_CPP_RPC=ON \
    -DUSE_RELAY_DEBUG=OFF \
    -DUSE_LIBBACKTRACE=OFF \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=armhf \
    -DCMAKE_C_COMPILER=arm-linux-gnueabihf-gcc \
    -DCMAKE_CXX_COMPILER=arm-linux-gnueabihf-g++ \
    || (cat CMakeFiles/CMakeError.log && false)



# Run make to build TVM
# RUN make -j2 || (cat CMakeFiles/CMakeError.log && false)
RUN make runtime -j$(nproc) || (cat CMakeFiles/CMakeError.log && false)

# Set environment variables
ENV TVM_HOME=/tvm
ENV PATH="$TVM_HOME/build:$PATH"
ENV PYTHONPATH="$TVM_HOME/python:$PYTHONPATH"
ENV LD_LIBRARY_PATH="$TVM_HOME/build:$LD_LIBRARY_PATH"



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
        Pillow==11.1.0 \
        entrypoints==0.4

# # # Package TVM build for easy transfer to Raspberry Pi
# # RUN tar -cf /tvm_rpi3.tar /tvm

# # # Entry point: Extract the build and set up TVM environment
# # CMD ["bash", "-c", "tar -xf /tvm_rpi3.tar -C / && bash"]