import torch
import pytorch.src.models as models
import pytorch.src.datasets as dataloaders
import tvm
from tvm import relay
from tvm.contrib import graph_executor
import numpy as np
import time
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='MobileNet')
    return parser.parse_args()

def decode_model_name(model_name):
    model_path = f"pytorch/model/{model_name}.pth"
    if model_name == "ResNet18":
        return models.ResNet(18, 10), model_path
    elif model_name == "ResNet34":
        return models.ResNet(34, 10), model_path
    elif model_name == "MobileNet":
        return models.MobileNet(10), model_path
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def compile_model(model, target):
    # convert to TorchScript
    example_input = torch.randn(1, 1, 28, 28)  # input shape: [batch_size, channel, height, width]
    traced_model = torch.jit.trace(model, example_input)

    # convert to Relay
    input_name = "input"
    input_shape = example_input.shape
    mod, params = relay.frontend.from_pytorch(traced_model, [(input_name, input_shape)])

    # compile
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)
    
    # Export the compiled model
    cross_compiler = "/usr/bin/arm-linux-gnueabihf-gcc"
    lib.export_library("tvm_rpi3.so", cc=cross_compiler)
    
    return lib

def main():
    args = parse_args()
    model_name = args.model
    target = args.target

    # load model
    model, model_path = decode_model_name(model_name)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    lib = compile_model(model, "llvm -mtriple=armv7l-linux-gnueabihf -mattr=+neon -mfloat-abi=hard")
    print(f"Model {model_name} is compiled successfully.")

if __name__ == "__main__":
    main()