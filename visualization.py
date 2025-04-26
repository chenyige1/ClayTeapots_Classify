import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image
import json
import os
import random
from torchvision import transforms
import argparse


def visualize_predictions(args):
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.model_name == 'resnet34':
        model = torchvision.models.resnet34(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 4)
    elif args.model_name == 'vgg19':
        model = torchvision.models.vgg19(pretrained=False)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = torch.nn.Linear(num_ftrs, 4)
    elif args.model_name == 'googlenet':
        model = torchvision.models.googlenet(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 4)

    model_path = args.model_path if args.model_path else f"best_{args.model_name}_model.pt"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # 加载标签映射
    with open('label_mapping.json', 'r') as f:
        idx_to_class = json.load(f)

    idx_to_class = {int(k): v for k, v in idx_to_class.items()}

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 反归一化用于显示图片
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )

    # 获取测试图片
    test_dir = os.path.join(args.data_dir, 'test')
    test_images = []

    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                test_images.append(os.path.join(root, file))

    # 随机选择9张图片进行可视化
    if len(test_images) > 9:
        test_images = random.sample(test_images, 9)

    # 创建3x3的子图
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.ravel()

    for idx, img_path in enumerate(test_images):
        if idx >= 9:
            break

        # 加载图片
        image = Image.open(img_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        # 预测
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = probabilities.max(1)

        # 显示图片
        display_image = np.array(image.resize((224, 224)))
        axes[idx].imshow(display_image)
        axes[idx].axis('off')

        # 添加预测结果和置信度
        pred_class = idx_to_class[predicted.item()]
        axes[idx].set_title(f'Pred: {pred_class}\nConf: {confidence.item():.2%}', fontsize=12)

    # 隐藏空的子图
    for idx in range(len(test_images), 9):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig('prediction_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 创建混淆矩阵可视化
    if args.confusion_matrix:
        create_confusion_matrix(model, test_dir, transform, idx_to_class, device)


def create_confusion_matrix(model, test_dir, transform, idx_to_class, device):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    true_labels = []
    pred_labels = []

    # 遍历测试集
    for class_idx, class_name in idx_to_class.items():
        class_dir = os.path.join(test_dir, class_name)
        if os.path.exists(class_dir):
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)

                    # 预测
                    image = Image.open(img_path).convert('RGB')
                    image_tensor = transform(image).unsqueeze(0).to(device)

                    with torch.no_grad():
                        outputs = model(image_tensor)
                        _, predicted = outputs.max(1)

                    true_labels.append(class_idx)
                    pred_labels.append(predicted.item())

    # 创建混淆矩阵
    cm = confusion_matrix(true_labels, pred_labels)

    # 可视化
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(idx_to_class.values()),
                yticklabels=list(idx_to_class.values()))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize Model Predictions')
    parser.add_argument('--data-dir', type=str, default='./classify', help='path to dataset')
    parser.add_argument('--model-name', type=str, default='resnet34', choices=['resnet34', 'vgg19', 'googlenet'],
                        help='model architecture')
    parser.add_argument('--model-path', type=str, default=None, help='path to the trained model')
    parser.add_argument('--confusion-matrix', action='store_true', help='generate confusion matrix')

    args = parser.parse_args()
    visualize_predictions(args)