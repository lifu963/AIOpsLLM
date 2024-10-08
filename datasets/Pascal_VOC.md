# Pascal VOC

## Dataset Summary

Pascal VOC（Visual Object Classes）是一个用于目标检测和图像分割任务的经典数据集。它包含多个子数据集，最常用的是VOC2007和VOC2012。该数据集涵盖了20个类别的目标，如人、狗、车等，并提供了目标位置和类别标注，以及图像分割的标注信息。Pascal VOC数据集在计算机视觉领域中被广泛用于算法性能评估、比较研究以及推动目标检测和图像分割技术的发展。

## Task

Object Detection

## Dataset Structure

### Data Instances

每个数据实例包含以下字段：

```json
{
  "image": "image tensor",
  "annotations": "list of annotation dictionaries"
}
```

### Data Fields

- image: 代表图像的张量，形状为 (3, H, W)，表示彩色图像，其中 H 和 W 分别是图像的高度和宽度。
- annotations: 包含了图像中目标的标注信息，每个标注信息以字典形式表示，包括以下字段：
  - "bbox": 目标边界框的坐标，格式为 [x_min, y_min, width, height]
  - "category_id": 目标类别的索引，对应 Pascal VOC 数据集中的类别
  - "segmentation": 目标的分割掩码，以多边形或分割区域的形式表示

## How to Use

```python
import torch
from torchvision import datasets, transforms

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),           # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
])

# Load Pascal VOC dataset
train_dataset = datasets.VOCDetection(root='./data', year='2007', image_set='train', download=True, transform=transform)
test_dataset = datasets.VOCDetection(root='./data', year='2007', image_set='test', download=True, transform=transform)

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Example usage:
for images, targets in train_loader:
    # Perform training iteration using images and targets
    pass
```

此处的代码演示了如何使用PyTorch加载Pascal VOC数据集，并创建用于训练和测试的数据加载器。首先，定义了数据转换，包括调整图像大小、转换为张量以及标准化。然后，使用`datasets.VOCDetection`类加载训练和测试数据集，并指定数据集的年份和子集。最后，创建了用于训练和测试的数据加载器，并可以在训练循环中使用它们进行模型训练。