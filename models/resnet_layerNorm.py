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

    def __init__(self, in_planes, planes, stride=1, batch_norm=True, layer_norm=False):
        super(BasicBlock, self).__init__()
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, 
            stride=stride, padding=1, bias=False
        )		if batch_norm:
			self.bn1 = nn.BatchNorm2d(planes)		if layer_norm:
			self.ln1 = nn.GroupNorm(1, planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3,
            stride=1, padding=1, bias=False
        )		if batch_norm:
			self.bn2 = nn.BatchNorm2d(planes)		if layer_norm:
			self.ln2 = nn.GroupNorm(1, planes)
        # self.lnorm2 = nn.LayerNorm([planes, 32, 32])

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion*planes,
                    kernel_size=1, stride=stride, bias=False
                ),				if batch_norm:
					nn.BatchNorm2d(self.expansion*planes),				if layer_norm:
					nn.GroupNorm(1, self.expansion*planes),
            )

    def forward(self, x):
        if self.batch_norm:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = F.relu(out)
        elif self.layer_norm:
            out = F.relu(self.ln1(self.conv1(x)))
            out = self.ln2(self.conv2(out))
            out += self.shortcut(x)
            out = F.relu(out)
        else:
            print(f'NormalizationChoiceError: Choose either BatchNorm or LayerNorm')
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, batch_norm=True, layer_norm=False):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3,
            stride=1, padding=1, bias=False
        )				if batch_norm:
			self.bn1 = nn.BatchNorm2d(64)		if layer_norm:
			self.ln1 = nn.GroupNorm(1, 64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, batch_norm=self.batch_norm, layer_norm=self.layer_norm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.batch_norm:
            out = F.relu(self.bn1(self.conv1(x)))
        elif self.layer_norm:
            out = F.relu(self.ln1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18_BatchNorm():
    return ResNet(BasicBlock, [2, 2, 2, 2],)

def ResNet18_LayerNorm():
    return ResNet(BasicBlock, [2, 2, 2, 2], batch_norm=False, layer_norm=True)


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())