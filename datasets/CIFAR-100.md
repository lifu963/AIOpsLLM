# CIFAR-100

## Dataset Summary

CIFAR-100 是一个经典的图像分类数据集，用于训练和评估图像分类模型。该数据集包含 100 个类别，每个类别包含 600 张尺寸为 32x32 的彩色图像。每个图像都有一个对应的类别标签，用于指示图像所属的类别。CIFAR-100 的图像类别涵盖了各种日常物体和场景，如动物、植物、交通工具、家具等，使其适用于多种应用场景的图像分类任务。

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

- image: 代表图像的张量，形状为 (3, 32, 32)，表示彩色图像，每个通道的尺寸为 32x32。
- label: 一个整数，表示图像所属的类别标签，范围从 0 到 99，对应 CIFAR-100 中的 100 个类别之一。

## How to Use

以下是使用 PyTorch 加载 CIFAR-100 数据集的示例代码：

```python
import torch
import torchvision
import torchvision.transforms as transforms

# 定义数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载训练集
trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

# 加载测试集
testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

# 类别标签
classes = trainset.classes

# 数据集信息
print('训练集数量：', len(trainset))
print('测试集数量：', len(testset))
print('类别数量：', len(classes))
print('类别标签：', classes)
```

此代码段首先使用 PyTorch 的 torchvision 模块加载 CIFAR-100 数据集，并定义了一系列的数据预处理操作，包括转换为张量和归一化。然后，数据集被分为训练集和测试集，并使用 DataLoader 加载到内存中。最后，打印了数据集的一些基本信息，如数据集大小、类别数量和类别标签。