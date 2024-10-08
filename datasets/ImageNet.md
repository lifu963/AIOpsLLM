# ImageNet

## Dataset Summary

ImageNet 是一个大规模的图像数据集，包含超过 100 万张高分辨率图像，涵盖了超过 1000 个类别。每个图像都有详细的类别标签，用于指示图像中所包含的物体或场景类别。ImageNet 数据集旨在提供一个丰富多样的图像数据集，以用于图像分类任务的训练和评估。由于其规模之大和类别之丰富，ImageNet 数据集成为了许多深度学习模型的标准评测基准之一。

## Task

Image Classification

## Dataset Structure

### Data Instances

每个数据实例包含以下字段：

```json
{
  "image": "image tensor",
  "label": "integer representing the class label"
}
```

### Data Fields

- image: 代表图像的张量，形状为 (3, H, W)，表示彩色图像，其中 H 和 W 分别是图像的高度和宽度。
- label: 一个整数，表示图像所属的类别标签，范围从 0 到 999，对应 ImageNet 数据集中的 1000 个类别之一。

## How to Use

以下是使用 PyTorch 加载 ImageNet 数据集的示例代码：

```python
import torch
import torchvision
import torchvision.transforms as transforms

# 定义数据预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载训练集
trainset = torchvision.datasets.ImageNet(root='./data', split='train',
                                          download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=8)

# 数据集信息
print('训练集数量：', len(trainset))
```

此代码段首先使用 PyTorch 的 torchvision 模块加载 ImageNet 数据集，并定义了一系列的数据预处理操作，包括调整大小、中心裁剪、转换为张量和归一化。然后，数据集被加载到 DataLoader 中，以便进行训练。