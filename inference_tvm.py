import tvm
from tvm.contrib import graph_executor
from torchvision import transforms
import numpy as np
from PIL import Image

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default="deploy.tar")
    parser.add_argument('-i', '--image', type=str, default="path/to/image.jpg")
    return parser.parse_args()

def main():
    args = parse_args()

    # Load the compiled model
    lib = tvm.runtime.load_module(args.model)

    dev = tvm.cpu()
    module = graph_executor.GraphModule(lib["default"](dev))

    transform = transforms.Compose([
            # transforms.Resize((28,28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    img = Image.open(args.image)
    data = transform(img)
    input_data = data.reshape(1, 1, 28, 28)
    input_data = np.array(input_data.numpy(), dtype="float32")
    module.set_input("input", input_data)
    module.run()
    output = module.get_output(0).asnumpy()

    print(f"Prediction: {np.argmax(output)}")

if __name__ == "__main__":
    main()