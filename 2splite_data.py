"""
Created on 2025/04/25
@author: cyg
实现将分类数据集按7 2 1比例放置
"""
import os
import shutil
import random
from pathlib import Path


def split_dataset(source_dir, target_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    将数据集按照指定比例分割成训练集、验证集和测试集

    Args:
        source_dir: 源数据集目录
        target_dir: 目标数据集目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
    """
    # 确保比例和为1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "比例之和必须为1"

    # 创建目标目录结构
    train_dir = os.path.join(target_dir, 'train')
    val_dir = os.path.join(target_dir, 'val')
    test_dir = os.path.join(target_dir, 'test')

    for dir_path in [train_dir, val_dir, test_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # 获取所有类别文件夹
    categories = [d for d in os.listdir(source_dir)
                  if os.path.isdir(os.path.join(source_dir, d))]

    print(f"找到类别: {categories}")

    # 处理每个类别
    for category in categories:
        source_category_dir = os.path.join(source_dir, category)

        # 创建目标类别文件夹
        train_category_dir = os.path.join(train_dir, category)
        val_category_dir = os.path.join(val_dir, category)
        test_category_dir = os.path.join(test_dir, category)

        for dir_path in [train_category_dir, val_category_dir, test_category_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # 获取所有图片文件
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp']:
            image_files.extend(Path(source_category_dir).glob(ext))
            image_files.extend(Path(source_category_dir).glob(ext.upper()))

        # 打乱文件顺序
        random.shuffle(image_files)

        # 计算分割点
        total_files = len(image_files)
        train_end = int(total_files * train_ratio)
        val_end = train_end + int(total_files * val_ratio)

        # 分割文件
        train_files = image_files[:train_end]
        val_files = image_files[train_end:val_end]
        test_files = image_files[val_end:]

        # 复制文件到目标目录
        for file_list, target_dir in [
            (train_files, train_category_dir),
            (val_files, val_category_dir),
            (test_files, test_category_dir)
        ]:
            for file in file_list:
                shutil.copy2(file, target_dir)

        print(
            f"类别 {category}: 总数={total_files}, 训练={len(train_files)}, 验证={len(val_files)}, 测试={len(test_files)}")


if __name__ == "__main__":
    # 使用示例
    source_dir = "./datasets"  # 原始数据集目录
    target_dir = "./split_datasets"  # 新的数据集目录

    split_dataset(source_dir, target_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
    print("数据集分割完成!")
