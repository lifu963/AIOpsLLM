# MNIST

## Dataset Summary

MNIST 是一个经典的手写数字图像数据集，用于机器学习领域中的图像分类任务。该数据集由 28x28 像素的灰度图像组成，共包含 0 到 9 十个类别，每个类别分别对应一个数字。MNIST 数据集最初由美国国家标准与技术研究所（NIST）创建，并在 Yann LeCun 等人的努力下成为了机器学习领域中的标准基准数据集。由于其规模适中、易于使用和广泛应用的特点，MNIST 数据集成为了入门级图像分类任务的理想选择，被广泛用于模型训练、算法验证和教学演示等领域。

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

- image: 代表图像的张量，形状为 (1, 28, 28)，表示灰度图像，高度和宽度均为 28 像素。
- label: 一个整数，表示图像所代表的数字，范围从 0 到 9，对应 MNIST 数据集中的 10 个类别之一。

## How to Use

以下是使用 PyTorch 加载 MNIST 数据集的示例代码：

```python
import torch
import torchvision
import torchvision.transforms as transforms

# 定义数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载训练集
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=2)

# 加载测试集
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=2)

# 数据集信息
print('训练集数量：', len(trainset))
print('测试集数量：', len(testset))
```

此代码段首先使用 PyTorch 的 torchvision 模块加载 MNIST 数据集，并定义了一系列的数据预处理操作，包括转换为张量和归一化。然后，数据集被分为训练集和测试集，并使用 DataLoader 加载到内存中。最后，打印了数据集的一些基本信息，如数据集大小、类别数量等。