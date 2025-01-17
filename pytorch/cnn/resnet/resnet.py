import torch.nn as nn
from torchsummary import summary


class Residual(nn.Module):
  def __init__(self, in_channels, num_channels, use_1conv: bool = False, stride: int = 1):
    super(Residual, self).__init__()
    self.ReLU = nn.ReLU()

    self.conv1 = nn.Conv2d(in_channels, num_channels, kernel_size=3, stride=stride, padding=1)
    self.bn1 = nn.BatchNorm2d(num_channels)
    self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
    self.bn2 = nn.BatchNorm2d(num_channels)
    if use_1conv:
      self.conv3 = nn.Conv2d(in_channels, num_channels, kernel_size=1, stride=stride)
    else:
      self.conv3 = None

  def forward(self, x):
    y = self.ReLU(self.bn1(self.conv1(x)))
    y = self.bn2(self.conv2(y))
    if self.conv3:
      x = self.conv3(x)
    y = self.ReLU(y + x)
    return y


class ResNet18(nn.Module):
  def __init__(self, in_channels: int = 3, num_classes: int = 5):
    super(ResNet18, self).__init__()
    self.b1 = nn.Sequential(
      nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    )
    self.b2 = nn.Sequential(
      Residual(64, 64, use_1conv=False, stride=1), Residual(64, 64, use_1conv=False, stride=1)
    )
    self.b3 = nn.Sequential(
      Residual(64, 128, use_1conv=True, stride=2), Residual(128, 128, use_1conv=False, stride=1)
    )
    self.b4 = nn.Sequential(
      Residual(128, 256, use_1conv=True, stride=2), Residual(256, 256, use_1conv=False, stride=1)
    )
    self.b5 = nn.Sequential(
      Residual(256, 512, use_1conv=True, stride=2), Residual(512, 512, use_1conv=False, stride=1)
    )
    self.b6 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(512, num_classes))

  def forward(self, x):
    y = self.b1(x)
    y = self.b2(y)
    y = self.b3(y)
    y = self.b4(y)
    y = self.b5(y)
    y = self.b6(y)
    return y


if __name__ == "__main__":
  model = ResNet18()  # .to(DEVICE)
  print(summary(model, (1, 224, 224)))

# class BasicBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self, in_planes, planes, stride=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion*planes)
#             )
#
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out
#
#
# class Bottleneck(nn.Module):
#     expansion = 4
#
#     def __init__(self, in_planes, planes, stride=1):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(self.expansion*planes)
#
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion*planes)
#             )
#
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = F.relu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out
#
#
# class ResNet(nn.Module):
#     def __init__(self, block, num_blocks, num_classes=10):
#         super(ResNet, self).__init__()
#         self.in_planes = 64
#
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
#         self.linear = nn.Linear(512*block.expansion, num_classes)
#
#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1]*(num_blocks-1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = F.avg_pool2d(out, 4)
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)
#         return out
#
#
# def ResNet18():
#     return ResNet(BasicBlock, [2, 2, 2, 2])
#
# def ResNet34():
#     return ResNet(BasicBlock, [3, 4, 6, 3])
#
# def ResNet50():
#     return ResNet(Bottleneck, [3, 4, 6, 3])
#
# def ResNet101():
#     return ResNet(Bottleneck, [3, 4, 23, 3])
#
# def ResNet152():
#     return ResNet(Bottleneck, [3, 8, 36, 3])
