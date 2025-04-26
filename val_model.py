import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import json
import shutil
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, roc_curve, auc, \
    precision_recall_curve
from sklearn.preprocessing import label_binarize


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


def evaluate_model(args):
    # 创建输出目录
    output_dir = os.path.join('output', 'model_effect')
    os.makedirs(output_dir, exist_ok=True)

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

    # 创建反向映射（类名到索引）
    class_to_idx = {v: k for k, v in label_mapping.items()}

    # 获取所有类别名称，按索引排序
    class_names = [label_mapping[i] for i in sorted(label_mapping.keys())]

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

    # 获取所有图像路径和真实标签
    image_paths = []
    true_labels = []

    if os.path.isdir(args.val_dir):
        for class_name in os.listdir(args.val_dir):
            class_dir = os.path.join(args.val_dir, class_name)
            if os.path.isdir(class_dir) and class_name in class_to_idx:
                for file in os.listdir(class_dir):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_paths.append(os.path.join(class_dir, file))
                        true_labels.append(class_to_idx[class_name])
    else:
        raise ValueError(f"Invalid validation directory: {args.val_dir}")

    if not image_paths:
        raise ValueError(f"No valid images found in {args.val_dir}")

    print(f"Found {len(image_paths)} images in validation set")

    # 存储每个图像的预测结果
    y_true = []  # 真实标签
    y_pred = []  # 预测标签
    y_scores = []  # 预测概率

    # 处理所有图像
    with torch.no_grad():
        for image_path, true_label in tqdm(zip(image_paths, true_labels), desc="Evaluating", total=len(image_paths)):
            try:
                # 处理图像
                image = process_image(image_path, transform).to(device)

                # 预测
                outputs = model(image)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                predicted_idx = torch.argmax(probabilities).item()

                y_true.append(true_label)
                y_pred.append(predicted_idx)
                y_scores.append(probabilities.cpu().numpy())

            except Exception as e:
                print(f"Error processing {image_path}: {e}")

    # 转换为numpy数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)

    # 计算指标
    print("\n---- Evaluation Results ----")

    # 准确度
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # 每个类别的精确度、召回率和F1分数
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)

    # 宏平均指标:对每个类别单独计算相应的指标，然后对所有类别的指标取平均值
    precision_avg = precision_score(y_true, y_pred, average='macro')
    recall_avg = recall_score(y_true, y_pred, average='macro')
    f1_avg = f1_score(y_true, y_pred, average='macro')

    print(f"Macro Precision: {precision_avg:.4f}")
    print(f"Macro Recall: {recall_avg:.4f}")
    print(f"Macro F1: {f1_avg:.4f}")

    # 保存每个类别的指标到CSV
    metrics_file = os.path.join(output_dir, "class_metrics.csv")
    with open(metrics_file, 'w') as f:
        f.write("Class,Precision,Recall,F1-score\n")
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name},{precision[i]:.4f},{recall[i]:.4f},{f1[i]:.4f}\n")
        f.write(f"Macro Average,{precision_avg:.4f},{recall_avg:.4f},{f1_avg:.4f}\n")

    print(f"\nClass metrics saved to {metrics_file}")

    # 绘制混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # 在混淆矩阵中显示数字
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # 保存混淆矩阵图像
    cm_file = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_file, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {cm_file}")
    plt.close()

    # 计算并绘制ROC曲线和AUC
    # 对真实标签进行二值化
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

    # 计算每个类别的ROC曲线和AUC
    plt.figure(figsize=(10, 8))

    # 存储AUC值
    auc_values = []

    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        auc_values.append(roc_auc)
        plt.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')

    # 计算平均AUC
    mean_auc = np.mean(auc_values)

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (Mean AUC = {mean_auc:.2f})')
    plt.legend(loc="lower right")

    # 保存ROC曲线
    roc_file = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(roc_file, dpi=300, bbox_inches='tight')
    print(f"ROC curve saved to {roc_file}")
    plt.close()

    # 绘制Precision-Recall曲线
    plt.figure(figsize=(10, 8))

    # 存储平均精确度值
    pr_auc_values = []

    for i in range(num_classes):
        precision_curve, recall_curve, _ = precision_recall_curve(y_true_bin[:, i], y_scores[:, i])
        # 计算PR曲线下面积
        pr_auc = auc(recall_curve, precision_curve)
        pr_auc_values.append(pr_auc)
        plt.plot(recall_curve, precision_curve, lw=2, label=f'{class_names[i]} (AUC = {pr_auc:.2f})')

    # 计算平均PR-AUC
    mean_pr_auc = np.mean(pr_auc_values)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (Mean AUC = {mean_pr_auc:.2f})')
    plt.legend(loc="lower left")

    # 保存PR曲线
    pr_file = os.path.join(output_dir, "precision_recall_curve.png")
    plt.savefig(pr_file, dpi=300, bbox_inches='tight')
    print(f"Precision-Recall curve saved to {pr_file}")
    plt.close()

    # 将指标保存为JSON文件
    metrics_json = {
        "accuracy": float(accuracy),
        "macro_precision": float(precision_avg),
        "macro_recall": float(recall_avg),
        "macro_f1": float(f1_avg),
        "mean_auc": float(mean_auc),
        "class_metrics": {
            class_names[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "auc": float(auc_values[i])
            } for i in range(num_classes)
        }
    }

    metrics_json_file = os.path.join(output_dir, "metrics.json")
    with open(metrics_json_file, 'w') as f:
        json.dump(metrics_json, f, indent=4)

    print(f"All metrics saved to {metrics_json_file}")

    # 绘制指标柱状图
    # 绘制每个类别的精确度、召回率和F1值
    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 8))
    bars1 = ax.bar(x - width, precision, width, label='Precision')
    bars2 = ax.bar(x, recall, width, label='Recall')
    bars3 = ax.bar(x + width, f1, width, label='F1-score')

    ax.set_xlabel('Classes')
    ax.set_ylabel('Scores')
    ax.set_title('Precision, Recall and F1-score by class')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45)
    ax.legend()

    # 在柱状图上添加数值标签
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    autolabel(bars1)
    autolabel(bars2)
    autolabel(bars3)

    plt.tight_layout()

    # 保存指标柱状图
    metrics_bar_file = os.path.join(output_dir, "metrics_by_class.png")
    plt.savefig(metrics_bar_file, dpi=300, bbox_inches='tight')
    print(f"Metrics bar chart saved to {metrics_bar_file}")
    plt.close()

    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model performance on validation set')
    parser.add_argument('--model-path', type=str, default='./output/best_resnet34_model.pt',
                        help='path to the trained model file')
    parser.add_argument('--val-dir', type=str, default='./split_datasets/val',
                        help='path to validation directory with class subdirectories')
    parser.add_argument('--use-gpu', action='store_true', help='use GPU if available')

    args = parser.parse_args()
    evaluate_model(args)