import torch
import torch.nn as nn
from torchvision import transforms, models
import os
import shutil
from PIL import Image
import json
import argparse
from tqdm import tqdm


# 获取模型
def get_model(model_name, num_classes=4):
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


# 图片分类函数
def classify_image(model, image_path, transform, device):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = outputs.max(1)

    return predicted.item()


# 主测试函数
def test_and_organize(args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载模型
    model = get_model(args.model_name, num_classes=4)
    model_path = args.model_path if args.model_path else f"best_{args.model_name}_model.pt"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # 加载标签映射
    with open('label_mapping.json', 'r') as f:
        idx_to_class = json.load(f)

    # 转换类型为整数到字符串的映射
    idx_to_class = {int(k): v for k, v in idx_to_class.items()}

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 创建输出目录
    output_dir = args.output_dir
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # 为每个类别创建子目录
    for class_name in idx_to_class.values():
        os.makedirs(os.path.join(output_dir, class_name))

    # 获取测试集图片
    test_dir = os.path.join(args.data_dir, 'test')
    test_images = []

    if os.path.exists(test_dir):
        for root, dirs, files in os.walk(test_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    test_images.append(os.path.join(root, file))

    # 分类并整理图片
    results = {}
    progress_bar = tqdm(test_images, desc='Classifying images')

    for img_path in progress_bar:
        try:
            # 预测类别
            predicted_idx = classify_image(model, img_path, transform, device)
            predicted_class = idx_to_class[predicted_idx]

            # 复制图片到对应类别文件夹
            dest_dir = os.path.join(output_dir, predicted_class)
            dest_path = os.path.join(dest_dir, os.path.basename(img_path))
            shutil.copy2(img_path, dest_path)

            # 记录结果
            results[img_path] = predicted_class

            progress_bar.set_postfix({'current_class': predicted_class})

        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")

    # 保存分类结果
    with open(os.path.join(output_dir, 'classification_results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    # 打印统计信息
    print("\nClassification Summary:")
    for idx, class_name in idx_to_class.items():
        count = len(os.listdir(os.path.join(output_dir, class_name)))
        print(f"{class_name}: {count} images")

    print(f"\nAll images have been classified and organized in: {output_dir}")
    print(f"Classification results saved to: {os.path.join(output_dir, 'classification_results.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image Classification Testing and Organization')
    parser.add_argument('--data-dir', type=str, default='./classify', help='path to dataset')
    parser.add_argument('--model-name', type=str, default='resnet34', choices=['resnet34', 'vgg19', 'googlenet'],
                        help='model architecture')
    parser.add_argument('--model-path', type=str, default=None, help='path to the trained model')
    parser.add_argument('--output-dir', type=str, default='./classified_results',
                        help='output directory for classified images')

    args = parser.parse_args()
    test_and_organize(args)