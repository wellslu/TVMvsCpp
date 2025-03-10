mkdir build
cd build
# cmake -DCMAKE_PREFIX_PATH=/Users/vincent/opt/anaconda3/envs/tvm/lib/python3.11/site-packages/torch/share/cmake ..
cmake -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..
cmake --build . --config Release