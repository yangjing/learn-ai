import copy
import time
import torch
from torch import nn, optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt


def load_train_datas():
  transform = transforms.Compose(
    [
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
    ]
  )
  train_data = torchvision.datasets.ImageFolder(root="./data/train", transform=transform)
  train_num = int(0.8 * len(train_data))
  train_data, val_data = random_split(train_data, [train_num, len(train_data) - train_num])
  train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2)
  val_loader = DataLoader(val_data, batch_size=32, shuffle=True, num_workers=2)

  return train_loader, val_loader
