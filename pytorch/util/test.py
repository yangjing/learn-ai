import time
import torch
import torch.nn as nn
from . import DEVICE


# 定义测试函数
def test(model, test_loader, DEVICE: torch.device = DEVICE):
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
