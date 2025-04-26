import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np


# 获取模型
def get_model(model_name, num_classes=6):
    if model_name == 'resnet34':
        model = models.resnet34(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'vgg19':
        model = models.vgg19(pretrained=False)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'googlenet':
        model = models.googlenet(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model


def process_image(image_path, transform):
    """处理图像为模型输入格式"""
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0), image  # 返回张量和原始图像


def classify_single_image(args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    print(f"Using device: {device}")

    # 加载标签映射
    label_mapping_path = os.path.join(os.path.dirname(args.model_path), 'label_mapping_resnet.json')
    with open(label_mapping_path, 'r') as f:
        label_mapping = json.load(f)

    # 将字符串键转换为整数
    label_mapping = {int(k): v for k, v in label_mapping.items()}
    num_classes = len(label_mapping)

    # 实例化模型
    model_name = os.path.basename(args.model_path).split('_')[1]  # 从文件名中提取模型名称
    model = get_model(model_name, num_classes=num_classes)

    # 加载模型权重
    model.load_state_dict(torch.load(args.model_path, map_location=device), strict=False)
    model = model.to(device)
    model.eval()

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 处理图像
    try:
        img_tensor, original_img = process_image(args.image_path, transform)
        img_tensor = img_tensor.to(device)

        # 预测
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

        # 获取每个类别的概率
        class_probs = {label_mapping[i]: prob.item() for i, prob in enumerate(probabilities)}
        predicted_idx = torch.argmax(probabilities).item()
        predicted_class = label_mapping[predicted_idx]
        confidence = probabilities[predicted_idx].item()

        # 尝试获取真实类别（如果根据目录结构推断）
        try:
            true_class = os.path.basename(os.path.dirname(args.image_path))
            if true_class in label_mapping.values():
                print(f"True class: {true_class}")
                is_correct = true_class == predicted_class
                print(f"Prediction {'correct' if is_correct else 'incorrect'}")
        except:
            pass

        # 获取模型对这个类别的整体准确率（如果有验证集结果）
        overall_accuracy = "None"  # 默认值

        # 可视化结果
        plt.figure(figsize=(12, 6))

        # 左边显示图像
        plt.subplot(1, 2, 1)
        plt.imshow(np.array(original_img))
        plt.title(f"Predicted: {predicted_class} ({confidence:.2%})")
        plt.axis('off')

        # 右边显示分类概率条形图
        plt.subplot(1, 2, 2)
        bars = plt.barh(list(class_probs.keys()), list(class_probs.values()), color='skyblue')
        plt.xlabel('Probability')
        plt.title('Class Probabilities')
        plt.xlim(0, 1)
        plt.grid(axis='x', linestyle='--', alpha=0.7)

        # 给最高置信度的条形图高亮
        bars[predicted_idx].set_color('orange')

        # 在条形图上添加概率值
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                     f'{class_probs[label_mapping[i]]:.2%}',
                     va='center')

        plt.tight_layout()

        # 保存可视化结果
        output_dir = os.path.join('output', 'visualize')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(args.image_path))[0]}_result.png")
        plt.savefig(output_path)

        print(f"Prediction: {predicted_class} with {confidence:.2%} confidence")
        print(f"Visualization saved to {output_path}")

        # 显示图像（如果非服务器环境）
        if not args.no_display:
            plt.show()

    except Exception as e:
        print(f"Error processing {args.image_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classify a single image with visualization')
    parser.add_argument('--model-path', type=str, default='./output/best_resnet34_model.pt', help='path to the trained model file')
    # parser.add_argument('--model-path', type=str, default='./output/best_googlenet_model.pt', help='path to the trained model file')
    # parser.add_argument('--image-path', type=str, default='./dataset/Flower_shape/IMG_20240628_143204.jpg', help='path to the image file to classify')
    parser.add_argument('--image-path', type=str, default='./round.png', help='path to the image file to classify')
    parser.add_argument('--use-gpu', action='store_true', help='use GPU if available')
    parser.add_argument('--no-display', action='store_true',
                        help='do not display visualization (for server environments)')

    args = parser.parse_args()
    classify_single_image(args)