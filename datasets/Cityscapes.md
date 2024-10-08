# Cityscapes

## Dataset Summary

Cityscapes是一个用于图像分割任务的数据集，包含来自城市街景的高分辨率图像。该数据集涵盖了各种天气条件、不同时间和季节的图像，并提供了精细的像素级标注，用于道路、车辆、行人等目标的分割。Cityscapes数据集旨在帮助研究人员和从业者开发和评估图像分割算法，以解决城市场景中的实际问题，如自动驾驶、交通监控等领域。

## Task

Image Segmentation

## Dataset Structure

### Data Instances

每个数据实例包含以下字段：

```json
{
  "image": "image tensor",
  "mask": "segmentation mask tensor"
}
```

### Data Fields

- image: 代表图像的张量，形状为 (3, H, W)，表示彩色图像，其中 H 和 W 分别是图像的高度和宽度。
- mask: 代表语义分割标注的张量，形状为 (H, W)，包含了对图像中每个像素的标签，用于区分不同的类别。

## How to Use

```python
import torch
from torchvision import datasets, transforms

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((512, 1024)),  # Resize images to 512x1024
    transforms.ToTensor(),           # Convert images to PyTorch tensors
])

# Load Cityscapes dataset
train_dataset = datasets.Cityscapes(root='./data', split='train', mode='fine', target_type='semantic', download=True, transform=transform)
val_dataset = datasets.Cityscapes(root='./data', split='val', mode='fine', target_type='semantic', download=True, transform=transform)
test_dataset = datasets.Cityscapes(root='./data', split='test', mode='fine', target_type='semantic', download=True, transform=transform)

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False)

# Example usage:
for images, targets in train_loader:
    # Perform training iteration using images and targets
    pass
```

此处的代码演示了如何使用PyTorch加载Cityscapes数据集，并创建用于训练、验证和测试的数据加载器。首先，定义了数据转换，包括调整图像大小和转换为张量。然后，使用`datasets.Cityscapes`类加载训练、验证和测试数据集，并指定了数据集的分割类型（semantic表示语义分割）。最后，创建了用于训练、验证和测试的数据加载器，并可以在训练循环中使用它们进行模型训练。