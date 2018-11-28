import torch
import torch.nn as nn

class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class EntryModule(nn.Module):
    def __init__(self):
        super(EntryModule, self).__init__()
        block1 = [  nn.Conv2d(3, 32, 3, stride=2),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, 3),
                    nn.BatchNorm2d(64),
                    nn.ReLU()]
        self.block1 = nn.Sequential(*block1)

        block2 = [  SeparableConv2d(64, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    SeparableConv2d(128, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.MaxPool2d(3, stride=1, padding=1)] # stride=2
        self.block2 = nn.Sequential(*block2)
        self.skip2 = nn.Conv2d(64, 128, 1) # stride=2

        block3 = [  SeparableConv2d(128, 256, 3, padding=1),
                    nn.BatchNorm2d(256),
                    SeparableConv2d(256, 256, 3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.MaxPool2d(3, stride=1, padding=1)] # stride=2
        self.block3 = nn.Sequential(*block3)
        self.skip3 = nn.Conv2d(128, 256, 1, stride=1) # stride=2

        block4 = [  SeparableConv2d(256, 728, 3, padding=1),
                    nn.BatchNorm2d(728),
                    SeparableConv2d(728, 728, 3, padding=1),
                    nn.BatchNorm2d(728),
                    nn.MaxPool2d(3, stride=1, padding=1)] # stride=2
        self.block4 = nn.Sequential(*block4)
        self.skip4 = nn.Conv2d(256, 728, 1, stride=1) # stride=2

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x) + self.skip2(x)
        x = self.block3(x) + self.skip3(x)
        x = self.block4(x) + self.skip4(x)
        return x

class MiddleModule(nn.Module):
    def __init__(self):
        super(MiddleModule, self).__init__()
        block1 = [   nn.ReLU(),
                    SeparableConv2d(728, 728, 3, padding=1),
                    nn.BatchNorm2d(728),
                    nn.ReLU(),
                    SeparableConv2d(728, 728, 3, padding=1),
                    nn.BatchNorm2d(728),
                    nn.ReLU(),
                    SeparableConv2d(728, 728, 3, padding=1),
                    nn.BatchNorm2d(728)]
        self.block1 = nn.Sequential(*block1)

        block2 = [   nn.ReLU(),
                    SeparableConv2d(728, 728, 3, padding=1),
                    nn.BatchNorm2d(728),
                    nn.ReLU(),
                    SeparableConv2d(728, 728, 3, padding=1),
                    nn.BatchNorm2d(728),
                    nn.ReLU(),
                    SeparableConv2d(728, 728, 3, padding=1),
                    nn.BatchNorm2d(728)]
        self.block2 = nn.Sequential(*block2)

        block3 = [   nn.ReLU(),
                    SeparableConv2d(728, 728, 3, padding=1),
                    nn.BatchNorm2d(728),
                    nn.ReLU(),
                    SeparableConv2d(728, 728, 3, padding=1),
                    nn.BatchNorm2d(728),
                    nn.ReLU(),
                    SeparableConv2d(728, 728, 3, padding=1),
                    nn.BatchNorm2d(728)]
        self.block3 = nn.Sequential(*block3)
    
    def forward(self, x):
        x = x + self.block1(x)
        x = x + self.block2(x)
        x = x + self.block3(x)
        return x

class ExitModule(nn.Module):
    def __init__(self):
        super(ExitModule, self).__init__()
        block1 = [  nn.ReLU(),
                    SeparableConv2d(728, 728, 3, padding=1),
                    nn.BatchNorm2d(728),
                    nn.ReLU(),
                    SeparableConv2d(728, 1024, 3, padding=1),
                    nn.BatchNorm2d(1024),
                    nn.MaxPool2d(3, stride=2)]
        self.block1 = nn.Sequential(*block1)
        self.skip1 = nn.Conv2d(728, 1024, 3, stride=2)

        block2 = [  SeparableConv2d(1024, 1536, 3, padding=1),
                    nn.BatchNorm2d(1536),
                    nn.ReLU(),
                    SeparableConv2d(1536, 2048, 3, padding=1),
                    nn.BatchNorm2d(2048),
                    nn.ReLU(),
                    nn.AvgPool2d(6)]
        self.block2 = nn.Sequential(*block2)

    def forward(self, x):
        x = self.skip1(x) + self.block1(x)
        x = self.block2(x)
        return x.view(x.size()[0], -1)

class Xception(nn.Module):
    def __init__(self):
        super(Xception, self).__init__()
        self.entry_module = EntryModule()
        self.middle_module = MiddleModule()
        self.exit_module = ExitModule()
        self.l1 = nn.Linear(2048, 256) # 2048
        self.l2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.entry_module(x)
        x = self.middle_module(x)
        x = self.exit_module(x)
        x = torch.relu(self.l1(x))
        x = self.l2(x)
        return x

    def embedding(self, x):
        x = self.entry_module(x)
        x = self.middle_module(x)
        x = self.exit_module(x)
        x = torch.relu(self.l1(x))
        return x