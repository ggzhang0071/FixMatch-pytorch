import os
import shutil
from pathlib import Path
import random

source_root = Path("/git/datasets")
target_root = Path("/git/fixmatch_dataset")

# 遍历所有子目录（以逗号分隔的多标签名）
all_images = []

for folder in source_root.iterdir():
    if folder.is_dir():
        labels = [label.strip() for label in folder.name.split(',')]
        for img_file in folder.glob("*.jpg"):
            all_images.append((img_file, labels))

# 随机打乱并划分
random.shuffle(all_images)
split_idx = int(0.8 * len(all_images))
train_data = all_images[:split_idx]
test_data = all_images[split_idx:]

# 工具函数：复制图像到所有标签目录
def copy_to_targets(data, subset: str):
    for img_path, labels in data:
        for label in labels:
            target_dir = target_root / subset / label
            target_dir.mkdir(parents=True, exist_ok=True)
            # 保持原文件名
            shutil.copy(img_path, target_dir / img_path.name)

copy_to_targets(train_data, "train")
copy_to_targets(test_data, "test")
