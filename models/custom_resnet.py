import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.MaxPool2d(, stride=2)
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out




class ResNet(nn.Module):
    def __init__(self,num_classes=10):
        super(ResNet, self).__init__()
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
                F.relu(),
                nn.Conv2d(128, 128,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(128),
                F.relu(),
            )
        self.layer2 = nn.Sequential(
                nn.Conv2d(128, 256,
                          kernel_size=3, stride=1, padding=1,bias=False),
                nn.MaxPool2d(2, stride=2),
                nn.BatchNorm2d(128)
            )
        self.layer3 = nn.Sequential(
                nn.Conv2d(256, 512,
                          kernel_size=3, stride=1, padding=1,bias=False),
                nn.MaxPool2d(2, stride=2),
                nn.BatchNorm2d(128)
            )
        self.resblk2 = nn.Sequential(
                nn.Conv2d(512, 512,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(128),
                F.relu(),
                nn.Conv2d(512, 512,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(128),
                F.relu(),
            )
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

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
        out = F.softmax(self.linear(out))
        return out


def ResNet18():
    return ResNet()


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])



def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())