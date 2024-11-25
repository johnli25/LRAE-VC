import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 56 * 56, 256),  # 假设输入为 224x224
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x



# 定义类别列表
classes = ['diving', 'golf_front', 'kick_front', 'lifting', 'riding_horse', 'running', 'skating', 'swing_bench']

def get_label_name(img_name):
    for name in classes:
        if name in img_name:
            return name
    raise ValueError("No corresponding class")

# 自定义 Dataset 类
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = os.listdir(img_dir)
        self.classes = classes
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        # 从文件名中提取类别
        label_name = get_label_name(img_name)
        label = self.class_to_idx[label_name]

        if self.transform:
            image = self.transform(image)

        return image, label

# data path, need to change to your own path
# input images path
train_val_dir = "UCF_224x224x3_PNC_FrameCorr_input_imgs/"  # 替换为训练和验证数据的路径
# decoded images path
test_dir = "output_test_imgs_post_training/"  # 替换为测试数据的路径

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 将图像调整到 224x224
    transforms.ToTensor(),         # 转换为张量
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化到 [-1, 1]
])

# 加载数据集
train_val_dataset = CustomImageDataset(img_dir=train_val_dir, transform=transform)
test_dataset = CustomImageDataset(img_dir=test_dir, transform=transform)

# 将 train & val 数据集划分为训练集和验证集
train_ratio = 0.8  # 80% 作为训练集，20% 作为验证集
train_size = int(len(train_val_dataset) * train_ratio)
val_size = len(train_val_dataset) - train_size
train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])

# 数据加载器
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 模型、损失函数、优化器
num_classes = len(classes)
model = SimpleCNN(num_classes).to(device)  # 将模型移动到 GPU（如果有）
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

# 准确率计算函数
def calculate_accuracy(loader, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)  # 数据移动到 GPU
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# 训练和验证
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)  # 数据移动到 GPU
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 验证集准确率
    val_accuracy = calculate_accuracy(val_loader, model)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Validation Accuracy: {val_accuracy:.2f}%")

# 测试集准确率
test_accuracy = calculate_accuracy(test_loader, model)
print(f"Test Accuracy: {test_accuracy:.2f}%")