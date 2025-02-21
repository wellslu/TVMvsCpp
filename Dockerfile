FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive 

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-setuptools python3-pip \
    llvm-9 llvm-9-dev clang-9 \
    cmake make g++ \
    libc6-dev-armhf-cross \
    qemu-user qemu-user-static \
    libtinfo-dev zlib1g-dev libedit-dev libxml2-dev \
    gcc-arm-linux-gnueabihf g++-arm-linux-gnueabihf \
    binutils-arm-linux-gnueabihf \
    git vim && rm -rf /var/lib/apt/lists/*

# Clone TVM
# ARG TVM_VERSION=main
# ARG GIT_URL=https://github.com/apache/tvm
# RUN git clone --recursive ${GIT_URL} /tvm
# WORKDIR /tvm
# RUN git checkout ${TVM_VERSION}
COPY tvm /tvm

# Prepare build directory
RUN mkdir -p /tvm/build
WORKDIR /tvm/build

# Cross-compile TVM for Raspberry Pi 3B (ARMv7)
RUN cmake -S .. -B . -DUSE_LLVM=ON -DUSE_MICRO=OFF -DUSE_RELAY=ERROR \
    -DLLVM_CONFIG=/usr/bin/llvm-config-9 \
    -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=armv7l \
    -DCMAKE_C_COMPILER=arm-linux-gnueabihf-gcc \
    -DCMAKE_CXX_COMPILER=arm-linux-gnueabihf-g++ \
    -DCMAKE_EXE_LINKER_FLAGS="-L/usr/arm-linux-gnueabihf/lib -lpthread" \
    -DCMAKE_SHARED_LINKER_FLAGS="-L/usr/arm-linux-gnueabihf/lib -lpthread" \
    -DTHREADS_PTHREAD_ARG="-pthread" \
    || (cat CMakeFiles/CMakeError.log && false)

# Run make to build TVM
RUN make -j$(nproc) || (cat CMakeFiles/CMakeError.log && false)

# Install Python dependencies
RUN pip3 --no-cache-dir install \
        cython \ 
        numpy==1.19.5 \
        torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
        opencv-python==4.6.0.66 \
        pyyaml==6.0.2 \
        tqdm==4.64.0 \
        scipy==1.5.4 \
        requests==2.32.3 \
        mlconfig==0.1.7 mlflow==1.28.0 \
        Pillow==11.1.0 \
        entrypoints==0.4

# Set environment variables
ENV TVM_HOME=/tvm
ENV PYTHONPATH=${TVM_HOME}/python:${PYTHONPATH}

# # Package TVM build for easy transfer to Raspberry Pi
# RUN tar -cf /tvm_rpi3.tar /tvm

# # Entry point: Extract the build and set up TVM environment
# CMD ["bash", "-c", "tar -xf /tvm_rpi3.tar -C / && bash"]