import torch
from PIL import Image
from torchvision import transforms

from pytorch.cnn.resnet.resnet import ResNet18
from pytorch.util import DATA_DIR

if __name__ == "__main__":
  model_file = "resnet-rice.pth"
  classes = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]
  image = Image.open(DATA_DIR / "Rice_Image_Dataset/test/Jasmine/Jasmine (14458).jpg")
  image2 = Image.open(DATA_DIR / "Rice_Image_Dataset/test/Basmati/basmati (1928).jpg")
  normalize = transforms.Normalize(
    [0.04207496, 0.04282031, 0.04414772], [0.03316495, 0.03434459, 0.0362894]
  )
  # 定义数据集处理方法变量
  transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
  images = torch.stack([transform(image), transform(image2)])

  # 添加批次维度

  model = ResNet18(in_channels=3, num_classes=5)  # .to(DEVICE)
  model.load_state_dict(torch.load(model_file, weights_only=True))

  with torch.no_grad():
    model.eval()
    images = images  # .to(DEVICE)
    output = model(images)
    pre_lab = torch.argmax(output, dim=1).cpu()

  # 打印预测结果
  for i, idx in enumerate(pre_lab):
    print(f"图像 {i + 1} 预测值：{classes[idx]}")
