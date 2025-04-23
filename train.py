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
def get_model(model_name, num_classes=4, pretrained=True):
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

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 创建模型
    model = get_model(args.model_name, num_classes=4, pretrained=True)
    model = model.to(device)

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
            torch.save(model.state_dict(), f"best_{args.model_name}_model.pt")
            print(f"Best model saved with accuracy: {best_val_acc * 100:.2f}%")

    # 保存最终模型
    torch.save(model.state_dict(), f"final_{args.model_name}_model.pt")
    print(f"Final model saved")

    # 保存标签映射
    import json
    label_mapping = train_dataset.idx_to_class
    with open('label_mapping.json', 'w') as f:
        json.dump(label_mapping, f)

    return model, label_mapping


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image Classification Training')
    parser.add_argument('--data-dir', type=str, default='./classify', help='path to dataset')
    parser.add_argument('--model-name', type=str, default='resnet34', choices=['resnet34', 'vgg19', 'googlenet'],
                        help='model architecture')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--num-workers', type=int, default=4, help='number of workers')

    args = parser.parse_args()
    train(args)