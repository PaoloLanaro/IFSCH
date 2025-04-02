import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Edit this to take in more than 28x28 images
class Conv_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        # 32 * 12 * 12 bc output of conv1 is 6 reducing the image from 28x28 to 24x24(28-5+1). 12x12 bc pool halves it.
        self.fc1 = nn.Linear(32*13*13, 500)
        self.fc2 = nn.Linear(500, 250)
        self.fc3 = nn.Linear(250,100)
        self.fc4 = nn.Linear(100, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        # flatten image
        x = torch.flatten(x, 1)
        # activation function with non linearality
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x