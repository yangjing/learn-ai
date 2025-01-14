import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from alexnet import AlexNet, DATA_DIR, DEVICE


def load_test_datas():
  # 加载测试数据集
  transform = transforms.Compose([transforms.Resize(size=227), transforms.ToTensor()])
  test_data = torchvision.datasets.FashionMNIST(
    root=DATA_DIR, train=False, download=True, transform=transform
  )
  test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=2)
  return test_loader


# 定义测试函数
def test(model: AlexNet, test_loader):
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
  # 加载训练好的模型
  model = AlexNet().to(DEVICE)
  model.load_state_dict(torch.load("alexnet.pth", weights_only=True))
  model.eval()

  test_loader = load_test_datas()
  # 运行测试函数
  test(model, test_loader)
