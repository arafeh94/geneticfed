import torch
import torchvision
from torch import nn
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

import torch.optim as optim
from libs.model.cv.cnn import Cifar10Model
from src import tools
from src.apis import lambdas
from src.data.data_loader import urls
from src.data.data_provider import PickleDataProvider

if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 50

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    data = PickleDataProvider(urls['cifar10']).collect().map(lambdas.reshape((32, 32, 3))).map(
        lambdas.transpose((2, 0, 1))).shuffle(47).as_tensor().split(0.8)
    train_data = data[0]
    test_data = data[1]

    net = Cifar10Model()
    tools.train(net, trainloader,lr=0.001)
    print(tools.infer(net, testloader))
    exit()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        # for i, data in enumerate(trainloader, 0):
        for i, data in enumerate(train_data.batch(50)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        # for data in testloader:
        for data in test_data.batch(50):
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))
    print('Finished Training')
