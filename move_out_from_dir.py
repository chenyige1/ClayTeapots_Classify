"""
Created on 2025/04/26
@author: cyg
实现将子文件夹下的图片移出到上一级目录
"""
import os
import shutil
from pathlib import Path


def flatten_directory(target_dir):
    """
    将指定目录下所有子文件夹中的文件移动到主文件夹，并删除空的子文件夹

    Args:
        target_dir: 目标目录
    """
    # 确保目标目录存在
    if not os.path.exists(target_dir):
        print(f"目录 {target_dir} 不存在")
        return

    # 获取所有子文件夹
    subdirs = [d for d in os.listdir(target_dir)
               if os.path.isdir(os.path.join(target_dir, d))]

    if not subdirs:
        print(f"没有找到子文件夹在 {target_dir}")
        return

    print(f"找到子文件夹: {subdirs}")

    # 处理每个子文件夹
    for subdir in subdirs:
        subdir_path = os.path.join(target_dir, subdir)

        # 获取子文件夹中的所有文件
        files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.*']:
            files.extend(Path(subdir_path).glob(ext))

        # 移动文件到主文件夹
        moved_count = 0
        for file in files:
            if file.is_file():
                # 处理重名文件
                target_file = os.path.join(target_dir, file.name)
                counter = 1

                while os.path.exists(target_file):
                    # 如果文件已存在，添加数字后缀
                    name, ext = os.path.splitext(file.name)
                    target_file = os.path.join(target_dir, f"{name}_{counter}{ext}")
                    counter += 1

                shutil.move(str(file), target_file)
                moved_count += 1

        print(f"从 {subdir} 移动了 {moved_count} 个文件")

        # 删除空文件夹
        if os.path.exists(subdir_path) and not os.listdir(subdir_path):
            os.rmdir(subdir_path)
            print(f"已删除空文件夹: {subdir}")


if __name__ == "__main__":
    # 使用示例
    # 移出train文件夹下的所有文件
    train_dir = "./split_datasets/train"
    flatten_directory(train_dir)

    # 移出val文件夹下的所有文件
    val_dir = "./split_datasets/val"
    flatten_directory(val_dir)

    # 移出test文件夹下的所有文件
    test_dir = "./split_datasets/test"
    flatten_directory(test_dir)

    print("文件移出完成!")