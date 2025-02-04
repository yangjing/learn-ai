import torch
from torch import nn
from torchsummary import summary
from pathlib import Path


DATA_DIR = Path.home() / "data"
DEVICE = torch.device(
  "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)


class LeNet(nn.Module):
  def __init__(self):
    super(LeNet, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
    self.sigmod = nn.Sigmoid()
    self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
    self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
    self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
    self.flatten = nn.Flatten()
    self.fc1 = nn.Linear(16 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    x = self.sigmod((self.conv1(x)))
    x = self.pool1(x)
    x = self.sigmod(self.conv2(x))
    x = self.pool2(x)
    x = self.flatten(x)
    x = self.fc1(x)
    x = self.fc2(x)
    x = self.fc3(x)
    return x


if __name__ == "__main__":
  device = torch.device(
    # "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    "cuda" if torch.cuda.is_available() else "cpu"
  )
  model = LeNet().to(device)
  print(summary(model, (1, 28, 28)))
