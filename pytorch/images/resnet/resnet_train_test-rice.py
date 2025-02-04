# 训练&测试: 稻米
import copy
import time
from os import PathLike
from typing import AnyStr

import torch
from torch import nn, optim
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import ImageFolder

from pytorch.images.resnet.resnet import ResNet18
from pytorch.util import DEVICE, DATA_DIR
import matplotlib.pyplot as plt


def load_datas(root: PathLike[AnyStr]):
  # 加载 FusionMNIST 数据集
  transform = transforms.Compose(
    [
      transforms.Resize(size=(224, 224)),
      transforms.ToTensor(),
      transforms.Normalize(
        [0.04207496, 0.04282031, 0.04414772], [0.03316495, 0.03434459, 0.0362894]
      ),
    ]
  )
  # 加载数据集
  train_data = ImageFolder(root / "train", transform=transform)
  train_num = int(0.8 * len(train_data))
  train_data, val_data = random_split(train_data, [train_num, len(train_data) - train_num])
  train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2)
  val_loader = DataLoader(val_data, batch_size=128, shuffle=True, num_workers=2)

  test_data = ImageFolder(root / "test", transform=transform)
  test_loader = DataLoader(test_data, batch_size=128, shuffle=True, num_workers=2)

  return train_loader, val_loader, test_loader


# 定义训练函数
def train(
  model, train_loader, val_loader, criterion, optimizer, epochs=5, model_file: str = "resnet.pth"
):
  # 复制当前模型的参数
  best_model_wts = copy.deepcopy(model.state_dict())
  best_acc = 0
  train_loss_epochs = []
  train_acc_epochs = []
  val_loss_epochs = []
  val_acc_epochs = []

  for epoch in range(epochs):
    start_time = time.time()
    train_loss = 0.0
    train_acc = 0.0

    model.train()
    for batch_x, batch_y in train_loader:
      batch_x = batch_x.to(DEVICE)
      batch_y = batch_y.to(DEVICE)
      optimizer.zero_grad()  # 将梯度置为零，防止将前一次的梯度累加

      outputs = model(batch_x)
      pred_y = torch.argmax(outputs, dim=1)  # 找到预测概率最大标签的索引下标
      loss = criterion(outputs, batch_y)

      loss.backward()  # 反向传播
      optimizer.step()  # 更新参数

      train_loss += loss.item()
      train_acc += torch.sum(pred_y == batch_y).cpu()

    train_loss /= len(train_loader)
    train_acc /= len(train_loader.dataset)
    train_loss_epochs.append(train_loss)
    train_acc_epochs.append(train_acc)

    val_loss = 0.0
    val_acc = 0.0
    model.eval()
    for batch_x, batch_y in val_loader:
      batch_x = batch_x.to(DEVICE)
      batch_y = batch_y.to(DEVICE)

      outputs = model(batch_x)
      pred_y = torch.argmax(outputs, dim=1)  # 找到预测概率最大标签的索引下标
      loss = criterion(outputs, batch_y)

      val_loss += loss.item()
      val_acc += torch.sum(pred_y == batch_y).cpu()

    val_loss /= len(val_loader)
    val_acc /= len(val_loader.dataset)
    val_loss_epochs.append(val_loss)
    val_acc_epochs.append(val_acc)

    end_time = time.time()  # 记录结束时间
    epoch_time = end_time - start_time  # 计算耗时
    print(
      f"Epoch {epoch + 1}|Time: {epoch_time:.3f}s - Train Loss: {train_loss:.3f} - Train Acc: {train_acc:.3f} - Val Loss: {val_loss:.3f} - Val Acc: {val_acc:.3f}"
    )

    if val_acc_epochs[epoch] > best_acc:
      best_acc = val_acc_epochs[epoch]
      best_model_wts = copy.deepcopy(model.state_dict())

  # 保存模型
  model.load_state_dict(best_model_wts)
  torch.save(model.state_dict(), model_file)

  print("Finished Training")

  train_res = {
    "epoch": range(1, epochs + 1),
    "train_loss_epochs": train_loss_epochs,
    "train_acc_epochs": train_acc_epochs,
    "val_loss_epochs": val_loss_epochs,
    "val_acc_epochs": val_acc_epochs,
  }
  return train_res


def matplot_acc_loss(res):
  plt.figure(figsize=(12, 12))

  plt.subplot(2, 1, 1)
  plt.plot(res["epoch"], res["train_loss_epochs"], "ro-", label="train loss")
  plt.plot(res["epoch"], res["val_loss_epochs"], "bs-", label="val loss")
  plt.legend()
  plt.xlabel("epoch")
  plt.ylabel("loss")

  plt.subplot(2, 1, 2)
  plt.plot(res["epoch"], res["train_acc_epochs"], "ro-", label="train acc")
  plt.plot(res["epoch"], res["val_acc_epochs"], "bs-", label="val acc")
  plt.legend()
  plt.xlabel("epoch")
  plt.ylabel("acc")

  plt.show()


def test(model: nn.Module, test_loader):
  test_loss = 0.0
  test_acc = 0.0
  criterion = nn.CrossEntropyLoss()

  with torch.no_grad():
    start_time = time.time()
    for batch_x, batch_y in test_loader:
      batch_x = batch_x.to(DEVICE)
      batch_y = batch_y.to(DEVICE)

      outputs = model(batch_x)
      pred_y = torch.argmax(outputs, dim=1)  # 找到预测概率最大标签的索引下标
      loss = criterion(outputs, batch_y)

      test_loss += loss.item()
      test_acc += torch.sum(pred_y == batch_y).cpu()

  test_loss /= len(test_loader)
  test_acc /= len(test_loader.dataset)
  end_time = time.time()  # 记录结束时间
  epoch_time = end_time - start_time  # 计算耗时
  print(f"Time: {epoch_time:.3f}s - Test Loss: {test_loss:.3f} - Test Acc: {test_acc:.3f}")


if __name__ == "__main__":
  train_loader, val_loader, test_loader = load_datas(DATA_DIR / "Rice_Image_Dataset")
  model_file = "resnet-rice.pth"

  # 运行训练
  net = ResNet18(in_channels=3, num_classes=5).to(DEVICE)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(net.parameters(), lr=0.001)
  train_res = train(
    net, train_loader, val_loader, criterion, optimizer, epochs=10, model_file=model_file
  )
  matplot_acc_loss(train_res)

  # 运行测试
  model = ResNet18(in_channels=3, num_classes=5).to(DEVICE)
  model.load_state_dict(torch.load(model_file, weights_only=True))
  model.eval()
  test(model, test_loader)
