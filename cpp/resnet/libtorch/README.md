# Steps

- Install PyTorch: refer to [torch install](https://pytorch.org/get-started/locally/).

- Get the model weights:
  - We provide an example in save_model_example.py to show how to export a PyTorch model that can be loaded by LibTorch.

- Set the model path **WEIGHTS_FOLDER** in main.cpp.
 
- Run build.sh to build our code:
    ```
    ./build.sh
    ```

- Run the executable:
    ```
    ./build/resnet
    ```