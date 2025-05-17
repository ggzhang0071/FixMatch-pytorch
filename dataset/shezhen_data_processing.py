import os
import shutil
from pathlib import Path
import random
import csv


# 这里定义一个函数保存标签csv文件
def save_labels_csv(data, csv_path):
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'labels'])  # header
        for img_path, labels in data:
            writer.writerow([img_path.name, ','.join(labels)])

def prepare_fixmatch_dataset(source_root, target_root, unlabeled_ratio=0.9):
    all_images = []
    for folder in source_root.iterdir():
        if folder.is_dir():
            labels = [label.strip() for label in folder.name.split(',')]
            for img_file in folder.glob("*.jpg"):
                all_images.append((img_file, labels))

    random.shuffle(all_images)

    split_idx = int(0.8 * len(all_images))
    train_data = all_images[:split_idx]
    test_data = all_images[split_idx:]

    train_labeled_num = int(len(train_data) * (1 - unlabeled_ratio))
    train_labeled = train_data[:train_labeled_num]
    train_unlabeled = train_data[train_labeled_num:]

    # 创建目录
    (target_root / 'train' / 'images').mkdir(parents=True, exist_ok=True)
    (target_root / 'train' / 'unlabeled').mkdir(parents=True, exist_ok=True)
    (target_root / 'test' / 'images').mkdir(parents=True, exist_ok=True)

    # 复制带标签的训练图像
    for img_path, _ in train_labeled:
        shutil.copy(img_path, target_root / 'train' / 'images' / img_path.name)

    # 保存带标签训练数据标签文件
    save_labels_csv(train_labeled, target_root / 'train' / 'labels.csv')

    # 复制无标签的训练图像
    for img_path, _ in train_unlabeled:
        shutil.copy(img_path, target_root / 'train' / 'unlabeled' / img_path.name)

    # 复制测试图像及标签
    for img_path, _ in test_data:
        shutil.copy(img_path, target_root / 'test' / 'images' / img_path.name)
    save_labels_csv(test_data, target_root / 'test' / 'labels.csv')
    print(f"Train labeled images: {len(train_labeled)}")
    print(f"Train unlabeled images: {len(train_unlabeled)}")
    print(f"Test images: {len(test_data)}")

if __name__ == "__main__":
    # 这里暂时只用了标签数据的地址，后续需要添加unlabeled数据的地址
    source_root = Path("/git/datasets/shezhen_original_data/shezhen_label_data")
    target_root = Path("/git/datasets/fixmatch_dataset")
    prepare_fixmatch_dataset(source_root, target_root, unlabeled_ratio=0.9)

