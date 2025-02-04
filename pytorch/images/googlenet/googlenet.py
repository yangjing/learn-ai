import torch
from torch import nn
from torchsummary import summary
from pathlib import Path


DATA_DIR = Path.home() / "data"
DEVICE = torch.device(
  "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)


class Inception(nn.Module):
  def __init__(self, in_channels: int, c1: int, c2: tuple[int, int], c3: tuple[int, int], c4: int):
    super(Inception, self).__init__()
    self.relu = nn.ReLU()

    # line1 1x1 conv
    self.p1_1 = nn.Conv2d(in_channels=in_channels, out_channels=c1, kernel_size=1)
    # line2 3x3 conv
    self.p2_1 = nn.Conv2d(in_channels=in_channels, out_channels=c2[0], kernel_size=1)
    self.p2_2 = nn.Conv2d(in_channels=c2[0], out_channels=c2[1], kernel_size=3, padding=1)
    # line3 5x5 conv
    self.p3_1 = nn.Conv2d(in_channels=in_channels, out_channels=c3[0], kernel_size=1)
    self.p3_2 = nn.Conv2d(in_channels=c3[0], out_channels=c3[1], kernel_size=5, padding=2)
    # line4 max pooling
    self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    self.p4_2 = nn.Conv2d(in_channels=in_channels, out_channels=c4, kernel_size=1)

  def forward(self, x):
    p1 = self.relu(self.p1_1(x))
    p2 = self.relu(self.p2_2(self.relu(self.p2_1(x))))
    p3 = self.relu(self.p3_2(self.relu(self.p3_1(x))))
    p4 = self.relu(self.p4_2(self.p4_1(x)))
    # (batch, channels, height, width) --> dim=1 --> channels 上拼接
    return torch.cat((p1, p2, p3, p4), dim=1)


class GoogLeNet(nn.Module):
  def __init__(self, in_channels: int = 1, num_classes: int = 10):
    super(GoogLeNet, self).__init__()

    self.b1 = nn.Sequential(
      nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    )
    self.b2 = nn.Sequential(
      nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    )
    self.b3 = nn.Sequential(
      Inception(192, 64, (96, 128), (16, 32), 32),
      Inception(256, 128, (128, 192), (32, 96), 64),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    )
    self.b4 = nn.Sequential(
      Inception(480, 192, (96, 208), (16, 48), 64),
      Inception(512, 160, (112, 224), (24, 64), 64),
      Inception(512, 128, (128, 256), (24, 64), 64),
      Inception(512, 112, (144, 288), (32, 64), 64),
      Inception(528, 256, (160, 320), (32, 128), 128),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    )
    self.b5 = nn.Sequential(
      Inception(832, 256, (160, 320), (32, 128), 128),
      Inception(832, 384, (192, 384), (48, 128), 128),
      nn.AdaptiveAvgPool2d((1, 1)),
      nn.Flatten(),
      nn.Linear(1024, num_classes),
    )

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
          nn.init.zeros_(m.bias)
      elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=0.01)
        if m.bias is not None:
          nn.init.zeros_(m.bias)

  def forward(self, x):
    x = self.b1(x)
    x = self.b2(x)
    x = self.b3(x)
    x = self.b4(x)
    x = self.b5(x)
    return x


if __name__ == "__main__":
  model = GoogLeNet()  # .to(DEVICE)
  print(summary(model, (1, 224, 224)))
