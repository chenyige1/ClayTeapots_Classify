import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse
import psutil
import platform


# 自定义数据集类
class ImageClassificationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.idx_to_class = {i: cls_name for i, cls_name in enumerate(self.classes)}

        self.images = []
        self.labels = []

        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(class_dir, img_name))
                    self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# 获取模型
def get_model(model_name, num_classes=6, pretrained=True):
    if model_name == 'resnet34':
        model = models.resnet34(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'vgg19':
        model = models.vgg19(pretrained=pretrained)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'googlenet':
        model = models.googlenet(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model


# 自动确定batch size
def determine_batch_size(model, device, model_name, test_batch_size=64):
    """
    自动确定最优的batch size
    """
    if device.type == 'cuda':
        # GPU模式下，根据显存确定batch size
        torch.cuda.empty_cache()
        total_memory = torch.cuda.get_device_properties(0).total_memory
        reserved_memory = torch.cuda.memory_reserved(0)
        allocated_memory = torch.cuda.memory_allocated(0)
        free_memory = total_memory - reserved_memory - allocated_memory

        # 根据模型大小预估batch size
        if model_name == 'resnet34':
            estimated_model_memory = 2 * 1024 * 1024 * 1024  # 约2GB
        elif model_name == 'vgg19':
            estimated_model_memory = 3 * 1024 * 1024 * 1024  # 约3GB
        else:  # googlenet
            estimated_model_memory = 1.5 * 1024 * 1024 * 1024  # 约1.5GB

        # 测试最大可行batch size
        test_input = torch.randn(test_batch_size, 3, 224, 224).to(device)
        try:
            with torch.no_grad():
                _ = model(test_input)

            # 逐步增加batch size，找到最大值
            while test_batch_size > 4:
                test_input = torch.randn(test_batch_size, 3, 224, 224).to(device)
                try:
                    with torch.no_grad():
                        _ = model(test_input)
                    return test_batch_size  # 如果成功，返回当前batch size
                except RuntimeError:
                    test_batch_size //= 2  # 如果失败，减半继续测试
                    torch.cuda.empty_cache()
        except RuntimeError:
            test_batch_size = 4  # 如果测试失败，使用最小值

    else:
        # CPU模式下，根据系统内存确定batch size
        available_memory = psutil.virtual_memory().available
        if available_memory > 16 * 1024 * 1024 * 1024:  # 16GB+
            test_batch_size = 32
        elif available_memory > 8 * 1024 * 1024 * 1024:  # 8GB+
            test_batch_size = 16
        elif available_memory > 4 * 1024 * 1024 * 1024:  # 4GB+
            test_batch_size = 8
        else:
            test_batch_size = 4

    return test_batch_size


# 确定最优的num_workers
def determine_num_workers():
    """
    根据CPU核心数确定最优的num_workers
    """
    num_cores = psutil.cpu_count(logical=False)  # 物理核心数
    if platform.system() == 'Windows':
        # Windows上多进程有一些限制，使用较少的workers
        return min(4, num_cores)
    else:
        # Linux/Mac上可以使用更多workers
        return min(8, num_cores)


# 训练函数
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc='Training')
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress_bar.set_postfix({
            'loss': f'{running_loss / (progress_bar.n + 1):.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })

    return running_loss / len(dataloader), correct / total


# 验证函数
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Validation')
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            progress_bar.set_postfix({
                'loss': f'{running_loss / (progress_bar.n + 1):.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

    return running_loss / len(dataloader), correct / total


# 主训练函数
def train(args):
    # 创建输出文件夹
    os.makedirs('output', exist_ok=True)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 数据预处理
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 创建数据集
    train_dataset = ImageClassificationDataset(os.path.join(args.data_dir, 'train'), transform=train_transform)
    val_dataset = ImageClassificationDataset(os.path.join(args.data_dir, 'val'), transform=val_transform)

    # 创建模型
    model = get_model(args.model_name, num_classes=6, pretrained=True)
    model = model.to(device)

    # 自动确定batch size
    if args.batch_size == -1:
        batch_size = determine_batch_size(model, device, args.model_name)
        print(f"Automatically determined batch size: {batch_size}")
    else:
        batch_size = args.batch_size

    # 自动确定num_workers
    if args.num_workers == -1:
        num_workers = determine_num_workers()
        print(f"Automatically determined num_workers: {num_workers}")
    else:
        num_workers = args.num_workers

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # 训练循环
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 10)

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc * 100:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc * 100:.2f}%")

        scheduler.step(val_loss)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join('output', f"best_{args.model_name}_model.pt"))
            print(f"Best model saved with accuracy: {best_val_acc * 100:.2f}%")

    # 保存最终模型
    torch.save(model.state_dict(), os.path.join('output', f"final_{args.model_name}_model.pt"))
    print(f"Final model saved")

    # 保存标签映射
    import json
    label_mapping = train_dataset.idx_to_class
    with open(os.path.join('output', 'label_mapping.json'), 'w') as f:
        json.dump(label_mapping, f)

    return model, label_mapping


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image Classification Training')
    parser.add_argument('--data-dir', type=str, default='./split_datasets', help='path to dataset')
    parser.add_argument('--model-name', type=str, default='googlenet', choices=['resnet34', 'vgg19', 'googlenet'],
                        help='model architecture')
    parser.add_argument('--batch-size', type=int, default=-1, help='batch size (-1 for auto)')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--num-workers', type=int, default=-1, help='number of workers (-1 for auto)')

    args = parser.parse_args()
    train(args)