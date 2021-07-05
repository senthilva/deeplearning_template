import torch
import torch.nn as nn
import torch.nn.functional as F


class custom_ResNet(nn.Module):
    def __init__(self,num_classes=10):
        super(custom_ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = nn.Sequential(
                nn.Conv2d(self.in_planes, 128,
                          kernel_size=3, stride=1, padding=1,bias=False),
                nn.MaxPool2d(2, stride=2),
                nn.BatchNorm2d(128)
            )
        self.resblk1 = nn.Sequential(
                nn.Conv2d(128, 128,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(),
            )
        self.layer2 = nn.Sequential(
                nn.Conv2d(128, 256,
                          kernel_size=3, stride=1, padding=1,bias=False),
                nn.MaxPool2d(2, stride=2),
                nn.BatchNorm2d(256)
            )
        self.layer3 = nn.Sequential(
                nn.Conv2d(256, 512,
                          kernel_size=3, stride=1, padding=1,bias=False),
                nn.MaxPool2d(2, stride=2),
                nn.BatchNorm2d(512)
            )
        self.resblk2 = nn.Sequential(
                nn.Conv2d(512, 512,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, 512,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(512),
               nn.ReLU(),
            )
        self.linear = nn.Linear(512, num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x))) # prep layer
        X = F.relu(self.layer1(out))
        R1 = self.resblk1(X)
        out = R1 + X
        out = self.layer2(out)
        X = self.layer3(out)
        R2 = self.resblk2(X)
        out = R2 + X
        out = F.max_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = F.softmax(self.linear(out),dim = 1)
        return out


def cust_ResNet18():
    return custom_ResNet()

