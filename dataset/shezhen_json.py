import os
import csv
import logging
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import sys
sys.path.append("/git/fixmatch")
from dataset.randaugment import RandAugmentMC
from dataset.cifar import CIFAR10SSL

logger = logging.getLogger(__name__)

# 多分辨率均值和标准差
MEAN_STD_MAP = {
    32: {
        'mean': (0.51222039, 0.34143419, 0.3458274),
        'std': (0.25696389, 0.17949953, 0.18240256)
    },
    224: {
        'mean': (0.51056955, 0.34046703, 0.34484451),
        'std': (0.26472661, 0.18532229, 0.18797784)
    }
}

ALL_CLASSES = ['气虚质', '阳虚质', '阴虚质', '痰湿质', '湿热质', '血瘀质', '气郁质', '特禀质', '平和质']

class SheZhenDataset(Dataset):
    def __init__(self, root, train=True, unlabeled=False, transform=None, image_size=224):
        self.root = root
        self.train = train
        self.unlabeled = unlabeled
        self.transform = transform
        self.image_size = image_size

        if train:
            if unlabeled:
                path_txt = os.path.join(root, "unlabeled_images.txt")
                if not os.path.exists(path_txt):
                    raise FileNotFoundError(f"缺少未标记图像路径文件：{path_txt}")
                with open(path_txt, "r") as f:
                    self.samples = [(os.path.abspath(line.strip()), []) 
                                    for line in f if line.strip().endswith(".jpg")]
                self.data = [p for p, _ in self.samples]
            else:
                self.data_dir = os.path.join(root, "train", "images")
                label_file = os.path.join(root, "train", "labels.csv")
                self.samples = self._load_labels(label_file)
                self.data = [os.path.join(self.data_dir, fname) for fname, _ in self.samples]
        else:
            self.data_dir = os.path.join(root, "test", "images")
            label_file = os.path.join(root, "test", "labels.csv")
            self.samples = self._load_labels(label_file)
            self.data = [os.path.join(self.data_dir, fname) for fname, _ in self.samples]

        self.targets = [self._encode_labels(lbls) for _, lbls in self.samples]

    def _load_labels(self, csv_path):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"找不到标签文件：{csv_path}")
        samples = []
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                samples.append((row['filename'], row['labels'].split(',')))
        return samples

    def _encode_labels(self, labels):
        return torch.tensor([1 if cls in labels else 0 for cls in ALL_CLASSES], dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path = self.data[index]
        labels = self.samples[index][1]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.warning(f"无法打开图像 {img_path}: {e}")
            image = Image.new('RGB', (self.image_size, self.image_size), (0, 0, 0))
            labels = []

        if self.transform:
            image = self.transform(image)

        return image, self._encode_labels(labels)


class TransformFixMatch(object):
    def __init__(self, mean, std, image_size=32):
        self.weak = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=image_size, padding=int(image_size * 0.125), padding_mode='reflect')
        ])
        self.strong = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=image_size, padding=int(image_size * 0.125), padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)
        ])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def __call__(self, x):
        return self.normalize(self.weak(x)), self.normalize(self.strong(x))


def get_shezhen9(args):
    image_size = args.image_size
    stats = MEAN_STD_MAP.get(image_size, MEAN_STD_MAP[32])
    mean, std = stats['mean'], stats['std']

    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=image_size, padding=int(image_size * 0.125), padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    transform_val = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    base_resize = transforms.Resize((image_size, image_size))
    to_tensor = transforms.ToTensor()
    resize_transform = transforms.Compose([base_resize, to_tensor])
    fixmatch_transform = TransformFixMatch(mean, std, image_size)

    train_labeled = SheZhenDataset(args.root, train=True, unlabeled=False, transform=resize_transform, image_size=image_size)
    train_unlabeled = SheZhenDataset(args.root, train=True, unlabeled=True, transform=resize_transform, image_size=image_size)
    test_dataset = SheZhenDataset(args.root, train=False, transform=resize_transform, image_size=image_size)

    return (
        CIFAR10SSL(train_labeled, train=True, transform=transform_labeled),
        CIFAR10SSL(train_unlabeled, train=True, transform=fixmatch_transform),
        CIFAR10SSL(test_dataset, train=False, transform=transform_val)
    )


# Debug CLI 主函数
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='SheZhen Dataset Loader for FixMatch')
    parser.add_argument('--root', type=str, default='/git/datasets/fixmatch_dataset')
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--dataset', type=str, default='shezhen')
    args = parser.parse_args()

    dataset_getter = {'shezhen': get_shezhen9}[args.dataset]
    train_l, train_u, test = dataset_getter(args)
    print(f"Train Labeled Dataset Size: {len(train_l)}")
    print(f"Train Unlabeled Dataset Size: {len(train_u)}")
    print(f"Test Dataset Size: {len(test)}")
