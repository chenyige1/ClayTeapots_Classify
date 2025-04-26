# ClayTeapots_Classify
紫砂壶分类数据集与分类相关代码
# 图片4分类深度学习项目使用说明

## 项目结构

```
./classify/
├── train/        # 训练集
│   ├── rose/     # 类别1的图片
│   ├── daisy/    # 类别2的图片
│   ├── .../      # 其他类别
│   └── .../      # 其他类别
├── val/          # 验证集（结构同上）
└── test/         # 测试集（结构同上）
```

## 环境要求

```bash
pip install torch torchvision pillow tqdm
```

## 使用方法

### 1. 训练模型

基本训练命令：
```bash
python train.py --model-name resnet34 --epochs 20 --batch-size 32
```

可选参数：
- `--model-name`: 选择模型（resnet34, vgg19, googlenet）
- `--data-dir`: 数据集路径（默认：./classify）
- `--batch-size`: 批次大小（默认：32）
- `--epochs`: 训练轮数（默认：20）
- `--lr`: 学习率（默认：0.001）
- `--num-workers`: 数据加载线程数（默认：4）

训练完成后会保存：
- `best_resnet34_model.pt`: 验证集上性能最好的模型
- `final_resnet34_model.pt`: 最后一轮的模型
- `label_mapping.json`: 类别索引与名称的映射

### 2. 测试和整理图片

基本测试命令：
```bash
python test.py --model-name resnet34
```

可选参数：
- `--model-name`: 选择模型（resnet34, vgg19, googlenet）
- `--data-dir`: 数据集路径（默认：./classify）
- `--model-path`: 模型文件路径（默认：best_模型名_model.pt）
- `--output-dir`: 输出目录（默认：./classified_results）

测试完成后：
- 所有测试图片会被分类并复制到相应的类别文件夹
- 生成`classification_results.json`记录每张图片的分类结果

## 代码特点

1. **多模型支持**：支持ResNet34、VGG19和GoogLeNet三种经典网络结构
2. **迁移学习**：使用预训练权重进行微调，提高训练效率
3. **数据增强**：包含随机翻转、旋转等增强操作
4. **进度显示**：使用tqdm库显示训练和测试进度
5. **GPU加速**：自动检测并使用CUDA加速
6. **自动保存**：保存最佳模型和最终模型
7. **智能分类**：自动根据目录结构识别类别

## 注意事项

1. 确保数据集目录结构正确
2. 图片格式支持：PNG、JPG、JPEG
3. 建议使用GPU训练以加快速度
4. 可以通过调整batch_size来适应不同的GPU内存
5. 模型会自动选择可用的设备（CPU或CUDA）

## 示例命令

使用VGG19训练：
```bash
python train.py --model-name vgg19 --epochs 30 --batch-size 16
```

使用训练好的模型进行测试：
```bash
python test.py --model-name vgg19 --model-path final_vgg19_model.pt --output-dir ./vgg19_results
```

## 性能提示

1. ResNet34通常在准确率和速度之间有很好的平衡
2. VGG19可能需要更多内存但特征提取能力强
3. GoogLeNet在计算效率方面表现优秀
4. 如果GPU内存不足，可以减小batch_size
5. 建议先用少量epoch测试，确认代码运行正常后再进行完整训练
