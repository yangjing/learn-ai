import torch
from torch import nn
from torchsummary import summary
from pathlib import Path


DATA_DIR = Path.home() / "data"
DEVICE = torch.device(
  "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)


class AlexNet(nn.Module):
  def __init__(self, num_classes: int = 10):
    super(AlexNet, self).__init__()
    self.features = nn.Sequential(
      nn.Conv2d(1, 96, kernel_size=11, stride=4),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.Conv2d(96, 256, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.Conv2d(256, 384, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(384, 384, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(384, 256, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=3, stride=2),
    )
    # self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
    self.classifier = nn.Sequential(
      nn.Linear(256 * 6 * 6, 4096),
      nn.ReLU(),
      nn.Dropout(p=0.5),
      nn.Linear(4096, 4096),
      nn.ReLU(),
      nn.Dropout(p=0.5),
      nn.Linear(4096, num_classes),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.features(x)
    # x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.classifier(x)
    return x


if __name__ == "__main__":
  model = AlexNet(num_classes=10)  # .to(DEVICE)
  print(summary(model, (1, 227, 227)))
