import torch
from torch import nn

class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(BasicBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.downsample = None
        if in_channel!=out_channel or stride!=1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 3, padding=1, stride=stride),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, input):
        x = self.block(input)
        x += input if not self.downsample else self.downsample(input)
        return x

class Resnet18(nn.Module):
    def __init__(self, nc):
        super(Resnet18, self).__init__()
        channels = [64, 128, 256, 512]
        strides = [1, 2, 2, 2]
        in_channel = 64
        modules = [nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
                   nn.BatchNorm2d(64),
                   nn.ReLU(inplace=True),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        for i in range(len(channels)):
            modules.append(self.make_layers(2, in_channel, channels[i], strides[i]))
            in_channel = channels[i]
        self.features = nn.Sequential(*modules)
        self.classifier = nn.Linear(512, nc)


    def forward(self, input):
        x = self.features(input)
        x = nn.functional.adaptive_avg_pool2d(x, (1,1))
        x = x.view(x.shape[0], x.shape[1])
        x = self.classifier(x)
        return x

    def make_layers(self, num, in_channel, out_channel, stride=1):
        blocks = []
        for i in range(num):
            if not i:
                blocks.append(BasicBlock(in_channel, out_channel, stride=stride))
            else:
                blocks.append(BasicBlock(out_channel, out_channel))
        return nn.Sequential(*blocks)