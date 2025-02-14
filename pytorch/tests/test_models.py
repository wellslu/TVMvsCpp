import unittest
import sys
import os

import torch
from src.models import ResNet

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestModels(unittest.TestCase):

    def test_resnet(self):
        m = ResNet(18, 10)
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            y = m(x)

        self.assertListEqual(list(y.size()), [1, 10])


if __name__ == '__main__':
    unittest.main()
