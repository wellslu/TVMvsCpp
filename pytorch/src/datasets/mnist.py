import mlconfig
import torch
from torch.utils import data
from torchvision import datasets, transforms
import os
import numpy as np
from PIL import Image

@mlconfig.register
class MNIST(data.DataLoader):

    def __init__(self, batch_size: int, train: bool, **kwargs):
        transform = transforms.Compose([
            # transforms.Resize((28,28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        if train:
            dataset = datasets.MNIST('./data/mnist', train=True, download=True, transform=transform)
        else:
            dataset = datasets.MNIST('./data/mnist', train=False, transform=transform)

        super(MNIST, self).__init__(dataset=dataset, batch_size=batch_size, shuffle=train, **kwargs)