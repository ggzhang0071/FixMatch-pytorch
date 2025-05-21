import os
import shutil
from pathlib import Path
import random
import csv
import numpy as np
import pandas as pd

# 使用支持中文的字体，例如 SimHei（黑体）
#matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置默认字体为黑体
#matplotlib.rcParams['axes.unicode_minus'] = False    # 正常显示负号


all_classes = ['气虚质', '阳虚质', '阴虚质', '痰湿质', '湿热质', '血瘀质', '气郁质', '平和质']

def encode_labels(labels):
    label_vector = [1 if cls in labels else 0 for cls in all_classes]
    return label_vector

def analyze_multilabel_imbalance(Y, label_names=None, plot=True):
    Y = np.array(Y)  
    num_samples, num_labels = Y.shape

    if label_names is None:
        label_names = [f"Label_{i}" for i in range(num_labels)]

    label_counts = Y.sum(axis=0)  # 每个标签的正样本数
    label_freq = label_counts / num_samples  # 标签频率
    label_neg_counts = num_samples - label_counts  # 负样本数
    pnr = label_counts / (label_neg_counts + 1e-9)  # 防止除以0
    imbalance_ratio = label_freq.max() / (label_freq + 1e-9)  # IR
    label_entropy = -label_freq * np.log2(label_freq + 1e-9) - (1 - label_freq) * np.log2(1 - label_freq + 1e-9)

    # 整理成 DataFrame 方便查看
    df = pd.DataFrame({
        "Label": label_names,
        "Positive Samples": label_counts.astype(int),
        "Frequency": label_freq,
        "PNR (Pos/Neg)": pnr,
        "Imbalance Ratio (IR)": imbalance_ratio,
        "Entropy": label_entropy
    }).sort_values(by="Frequency", ascending=False)

    print(df.to_string(index=False))

   

# 这里定义一个函数保存标签csv文件
def save_labels_csv(data, csv_path):
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'labels'])  # header
        for img_path, labels in data:
            writer.writerow([img_path.name, ','.join(labels)])
def prepare_fixmatch_dataset(label_source_root, unlabeled_source_root, target_root, unlabeled_ratio=0.9):
    all_images = []
    all_labels = []
    for folder in label_source_root.iterdir():
        if folder.is_dir():
            labels = [label.strip() for label in folder.name.split(',')]
            for img_file in folder.glob("*.jpg"):
                all_images.append((img_file, labels))
                all_labels.append(encode_labels(labels)) 

    random.shuffle(all_images) 

    # 读取无标签数据（但不复制）
    unlabeled_images = []
    for img_file in unlabeled_source_root.rglob("*.jpg"):
        unlabeled_images.append((img_file, []))

    split_idx = int(0.8 * len(all_images))
    train_labeled = all_images[:split_idx]
    test_data = all_images[split_idx:]

    # 创建目录
    (target_root / 'train' / 'images').mkdir(parents=True, exist_ok=True)
    (target_root / 'test' / 'images').mkdir(parents=True, exist_ok=True)

    # 复制有标签训练图像
    for img_path, _ in train_labeled:
        shutil.copy(img_path, target_root / 'train' / 'images' / img_path.name)

    # 保存训练标签
    save_labels_csv(train_labeled, target_root / 'train' / 'labels.csv')

    # 测试集
    for img_path, _ in test_data:
        shutil.copy(img_path, target_root / 'test' / 'images' / img_path.name)
    save_labels_csv(test_data, target_root / 'test' / 'labels.csv')

    # 保存无标签图像路径引用（而不是复制图像）
    with open(target_root / "unlabeled_path.txt", "w") as f:
        f.write(str(unlabeled_source_root.resolve()))

    print(f"Train labeled images: {len(train_labeled)}")
    print(f"Unlabeled images directory remains at: {unlabeled_source_root}")
    print(f"Found {len(unlabeled_images)} unlabeled images.")

    print(f"Test images: {len(test_data)}")


if __name__ == "__main__":
    # 这里暂时只用了标签数据的地址，后续需要添加unlabeled数据的地址
    label_source_root = Path("/git/datasets/shezhen_original_data/shezhen_label_data")
    unlabeled_source_root = Path("/git/datasets/shezhen_original_data/shezhen_unlabel_data")
    target_root = Path("/git/datasets/fixmatch_dataset")
    if target_root.exists():
        shutil.rmtree(target_root)
    prepare_fixmatch_dataset(label_source_root,unlabeled_source_root, target_root, unlabeled_ratio=0.5)

