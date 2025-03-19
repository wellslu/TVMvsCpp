# Steps:

- Get the model weights:
    - Use pytorch/convert_pth_2_npz.py to convert the model weights exported from PyTorch to a npz data that can be loaded by the C++ program.

- Install cnpy:
    - Build [cnpy](https://github.com/rogersce/cnpy/tree/master) from source code.

    - Update LD_LIBRARY_PATH so that the programs can find the libcnpy.so.

- Install OpenCV
    ```bash
    sudo apt update

    # install dependacies
    sudo apt install build-essential cmake git pkg-config libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libjasper-dev libatlas-base-dev gfortran python3-dev

    # install opencv
    sudo apt install libopencv-dev
    ```
- Set the model path **WEIGHTS_FOLDER** and the **imagePath** in main.cpp.
- Compile the source code:
    ```bash
    make
    ```
- Run the executable:
    ```
    ./resnet
    ```