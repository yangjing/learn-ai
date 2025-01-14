import copy
import time
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from . import DEVICE


# 定义训练函数
def train(
  model, train_loader, val_loader, criterion, optimizer, epochs=5, DEVICE: torch.device = DEVICE
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
  torch.save(model.state_dict(), "alexnet.pth")

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
