import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import json
import shutil
import argparse
from tqdm import tqdm


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
    return transform(image).unsqueeze(0)  # 添加批次维度


def validate_dataset(args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    print(f"Using device: {device}")

    # 加载标签映射
    label_mapping_path = os.path.join(os.path.dirname(args.model_path), './label_mapping_resnet.json')
    with open(label_mapping_path, 'r') as f:
        label_mapping = json.load(f)

    # 将字符串键转换为整数
    label_mapping = {int(k): v for k, v in label_mapping.items()}
    num_classes = len(label_mapping)
    print(f"Loaded label mapping with {num_classes} classes")

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

    # 创建输出目录
    output_dir = os.path.join('output', 'classify')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # 为每个类别创建目录
    for class_name in label_mapping.values():
        os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)

    # 统计每个类别的预测情况
    class_metrics = {class_name: {'correct': 0, 'total': 0} for class_name in label_mapping.values()}

    # 获取所有图像路径
    image_paths = []
    if os.path.isdir(args.data_dir):
        for root, _, files in os.walk(args.data_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(root, file))
    else:
        raise ValueError(f"Invalid data directory: {args.data_dir}")

    print(f"Found {len(image_paths)} images to process")

    # 处理所有图像
    with torch.no_grad():
        for image_path in tqdm(image_paths, desc="Processing images"):
            try:
                # 提取真实类别（如果根据目录结构推断）
                try:
                    true_class = os.path.basename(os.path.dirname(image_path))
                except:
                    true_class = None

                # 处理图像
                image = process_image(image_path, transform).to(device)

                # 预测
                outputs = model(image)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                predicted_idx = torch.argmax(probabilities).item()
                confidence = probabilities[predicted_idx].item()

                predicted_class = label_mapping[predicted_idx]

                # 更新统计信息
                if true_class in label_mapping.values():
                    class_metrics[true_class]['total'] += 1
                    if true_class == predicted_class:
                        class_metrics[true_class]['correct'] += 1

                # 复制图像到对应类别目录
                filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_{confidence:.2f}{os.path.splitext(image_path)[1]}"
                dest_path = os.path.join(output_dir, predicted_class, filename)
                shutil.copy2(image_path, dest_path)

            except Exception as e:
                print(f"Error processing {image_path}: {e}")

    # 打印分类结果统计
    print("\n---- Classification Results ----")
    for class_name, metrics in class_metrics.items():
        if metrics['total'] > 0:
            accuracy = metrics['correct'] / metrics['total'] * 100
            print(f"{class_name}: {metrics['correct']}/{metrics['total']} ({accuracy:.2f}%)")

    print(f"\nClassified images saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate and classify images with a trained model')
    parser.add_argument('--model-path', type=str, default='./output/best_googlenet_model.pt', help='path to the trained model file')
    parser.add_argument('--data-dir', type=str, default='./split_datasets/test', help='path to image directory to classify')
    parser.add_argument('--use-gpu', action='store_true', help='use GPU if available')

    args = parser.parse_args()
    validate_dataset(args)