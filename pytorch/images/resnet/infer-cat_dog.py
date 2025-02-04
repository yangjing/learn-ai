import torch
from PIL import Image
from torchvision import transforms

from pytorch.images.resnet.resnet import ResNet18
from pytorch.util import DATA_DIR

if __name__ == "__main__":
  model_file = "resnet-cat_dog.pth"
  classes = ["Cat", "Dog"]
  image = Image.open(DATA_DIR / "cat_dog/test/cat/3655.jpg")
  image2 = Image.open(DATA_DIR / "cat_dog/test/dog/2145.jpg")
  normalize = transforms.Normalize(
    [0.16207108, 0.15101928, 0.13847153], [0.05800501, 0.05212834, 0.04776142]
  )
  # 定义数据集处理方法变量
  transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
  images = torch.stack([transform(image), transform(image2)])

  # 添加批次维度

  model = ResNet18(in_channels=3, num_classes=2)  # .to(DEVICE)
  model.load_state_dict(torch.load(model_file, weights_only=True))

  with torch.no_grad():
    model.eval()
    images = images  # .to(DEVICE)
    output = model(images)
    pre_lab = torch.argmax(output, dim=1).cpu()

  # 打印预测结果
  for i, idx in enumerate(pre_lab):
    print(f"图像 {i + 1} 预测值：{classes[idx]}")
