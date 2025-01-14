import torch
from torch import nn
from torchsummary import summary
from pathlib import Path
import os


# 使用环境变量定义数据目录
DATA_DIR = Path(os.getenv("DATA_DIR", str(Path.home() / "data")))
DEVICE = torch.device(
  "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)


class VGG16(nn.Module):
  def __init__(self, num_classes: int = 1000, input_channels: int = 1):
    super(VGG16, self).__init__()
    self.block1 = nn.Sequential(
      nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(64, 64, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
    )
    self.block2 = nn.Sequential(
      nn.Conv2d(64, 128, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(128, 128, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
    )
    self.block3 = nn.Sequential(
      nn.Conv2d(128, 256, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(256, 256, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(256, 256, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
    )
    self.block4 = nn.Sequential(
      nn.Conv2d(256, 512, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(512, 512, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(512, 512, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
    )
    self.block5 = nn.Sequential(
      nn.Conv2d(512, 512, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(512, 512, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(512, 512, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
    )
    self.classifier = nn.Sequential(
      nn.Flatten(),  # 展平张量，类似于 torch.flatten(x, 1)
      nn.Linear(512 * 7 * 7, 4096),
      nn.ReLU(),
      # nn.Dropout(),
      nn.Linear(4096, 4096),
      nn.ReLU(),
      # nn.Dropout(),
      nn.Linear(4096, num_classes),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # 验证输入张量形状
    expected_shape = (x.size(0), 1, 224, 224)
    if x.shape != expected_shape:
      raise ValueError(f"Expected input shape {expected_shape}, but got {x.shape}")

    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    x = self.block4(x)
    x = self.block5(x)
    x = self.classifier(x)
    return x


if __name__ == "__main__":
  model = VGG16(num_classes=10)  # .to(DEVICE)
  print(summary(model, (1, 224, 224)))
