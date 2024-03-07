import torch
from torch import nn

from apps.splitfed.models import MnistNet
from src.apis import federated_tools
from src.data.data_distributor import ShardDistributor
from src.data.data_loader import preload


class CNN2DModel(nn.Module):
    def __init__(self):
        super(CNN2DModel, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 14 * 14, 128)  # Adjust input size based on your image dimensions
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)  # Output size is 10 for the 10 digits

    def forward(self, x):
        # Forward pass through convolutional layers
        x.shape(-1)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool(x)

        # Flatten the output for fully connected layers
        x = x.view(-1, 64 * 14 * 14)  # Adjust based on your image dimensions

        # Forward pass through fully connected layers
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)

        return x


original_image_shape = (28, 28)

mnist = preload('mnist').select(range(1)).reshape(original_image_shape).as_tensor()
model = CNN2DModel()
federated_tools.train(model, mnist.batch())
