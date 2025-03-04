import tvm
from tvm.contrib import graph_executor
from torchvision import transforms
import numpy as np
from PIL import Image
import time

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default="tvm_x86.so")
    parser.add_argument('-i', '--image', type=str, default="3.png")
    return parser.parse_args()

def main():
    args = parse_args()

    # Load the compiled model
    lib = tvm.runtime.load_module(args.model)

    dev = tvm.cpu()
    module = graph_executor.GraphModule(lib["default"](dev))

    transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28,28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    start = time.time()

    for i in range(1000):
        img = Image.open(args.image)
        data = transform(img)
        input_data = data.reshape(1, 1, 28, 28)
        input_data = np.array(input_data.numpy(), dtype="float32")
        module.set_input("input", input_data)
        module.run()
        output = module.get_output(0).asnumpy()

    print(f"Inference 1000 times with reload the same image: {time.time() - start}")

if __name__ == "__main__":
    main()