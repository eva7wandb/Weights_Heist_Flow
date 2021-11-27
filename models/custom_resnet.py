'''ResNet in PyTorch.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, 
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        # self.shortcut = nn.Sequential()
        # if stride != 1 or in_planes != self.expansion*planes:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(
        #             in_planes, self.expansion*planes,
        #             kernel_size=1, stride=stride, bias=False
        #         ),
        #         nn.BatchNorm2d(self.expansion*planes)
        #     )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # out += self.shortcut(x)
        out = F.relu(out)
        return out

class XBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(XBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, 
            stride=stride, padding=1, bias=False
        )
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.bn1 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = self.bn1(self.maxpool1(self.conv1(x)))
        out = F.relu(out)
        return out

class CustResNet(nn.Module):
    def __init__(self, block, xblock, num_blocks, num_classes=10):
        super(CustResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1_x = self._make_layer(xblock, 128, num_blocks[0], stride=1)
        self.layer1_R1 = self._make_layer(block, 128, num_blocks[0], stride=1)

        self.layer2 = self._make_layer(xblock, 256, num_blocks[1], stride=1)

        self.layer3_x = self._make_layer(xblock, 512, num_blocks[2], stride=1)
        self.layer3_R2 = self._make_layer(block, 512, num_blocks[2], stride=1)

        #self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        #prep layer
        out = F.relu(self.bn1(self.conv1(x)))
        #layer 1
        out = self.layer1_x(out)
        out = out + self.layer1_R1(out)
        #layer 2
        out = self.layer2(out)
        #layer 3
        out = self.layer3_x(out)
        out = out + self.layer3_R2(out) 
        #out = self.layer4(out)
        out = F.max_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return F.log_softmax(out, dim=-1)


def CustomResNet():
    return CustResNet(BasicBlock, XBlock, [1, 1, 1])

def test():
    net = CustomResNet()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())