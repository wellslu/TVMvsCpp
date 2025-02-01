import mlconfig
import torch
from torch.utils import data
from torchvision import datasets, transforms
import numpy as np
from PIL import Image

@mlconfig.register
class CIFAR10(data.DataLoader):

    def __init__(self, batch_size: int, train: bool, **kwargs):
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        if train:
            dataset = datasets.CIFAR10('./data/cifar10', train=True, download=True, transform=transform)
        else:
            dataset = datasets.CIFAR10('./data/cifar10', train=False, transform=transform)
        
        # classes = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

        super(CIFAR10, self).__init__(dataset=dataset, batch_size=batch_size, shuffle=train, **kwargs)