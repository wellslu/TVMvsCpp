import torch
import pytorch.src.models as models
from torchvision import transforms
import numpy as np
from PIL import Image
import time
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='MobileNet')
    parser.add_argument('-i', '--image', type=str, default="3.png")
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

def main():
    args = parse_args()

    args = parse_args()
    model_name = args.model

    # load model
    model, model_path = decode_model_name(model_name)
    model.load_state_dict(torch.load(model_path))
    model.eval()

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
        output = model(input_data)

    print(f"Inference 1000 times with reload the same image: {time.time() - start}")

if __name__ == "__main__":
    main()