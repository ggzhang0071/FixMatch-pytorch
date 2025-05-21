import logging
import math
import os
import csv
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from .randaugment import RandAugmentMC
from dataset.cifar import CIFAR10SSL

logger = logging.getLogger(__name__)



shezhen_32_mean=(0.51222039, 0.34143419, 0.3458274)
shezhen_32_std=(0.25696389, 0.17949953, 0.18240256)

shezhen_224_mean=(0.51056955, 0.34046703,0.34484451)
Shezhen_224_std=(0.26472661,0.18532229, 0.18797784)

normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)



class SheZhenDataset(Dataset):
    def __init__(self, root, train=True, unlabeled=False, transform=None,image_size=image_size):
        self.root = root
        self.train = train
        self.unlabeled = unlabeled 
        self.transform = transform
        self.all_classes = ['气虚质', '阳虚质', '阴虚质', '痰湿质', '湿热质', '血瘀质', '气郁质', '特禀质', '平和质']

        if train:
            if unlabeled:
                # 无标签数据使用原始地址
                self.data_dir = self.root
                #self.data_dir = os.path.join(root, 'train', 'unlabeled')
                self.samples = [(fname, []) for fname in os.listdir(self.data_dir)]

            else:
                self.data_dir = os.path.join(root, 'train', 'images')
                label_file = os.path.join(root, 'train', 'labels.csv')
                self.samples = self._load_labels(label_file)
        else:
            self.data_dir = os.path.join(root, 'test', 'images')
            label_file = os.path.join(root, 'test', 'labels.csv')
            self.samples = self._load_labels(label_file)
        
        self.data = [os.path.join(self.data_dir, fname) for fname, _ in self.samples]
        self.targets = [self._encode_labels(labels) for _, labels in self.samples]

    def _load_labels(self, csv_path):
        samples = []
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row['filename']
                labels = row['labels'].split(',')
                samples.append((fname, labels))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        fname, labels = self.samples[index]
        img_path = os.path.join(self.data_dir, fname)
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"[Warning] Cannot open image {img_path}: {e}")
            image = Image.new('RGB', (image_size, image_size), (0, 0, 0))
            labels = []

        if self.transform:
            image = self.transform(image)

        label_vector = self._encode_labels(labels)
        return image, label_vector

    def _encode_labels(self, labels):
        label_vector = [1 if cls in labels else 0 for cls in self.all_classes]
        return torch.tensor(label_vector, dtype=torch.float32)  




def get_shezhen9(args):
    label_root = args.label_root
    unlabeled_root = args.unlabeled_root
    image_size = args.image_size
    transform_labeled = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(size=image_size, padding=int(image_size*0.125), padding_mode='reflect'),
    transforms.ToTensor(),
    transforms.Normalize(mean=shezhen_32_mean, std=shezhen_32_std)
    ])

    # 测试/验证集 transform
    transform_val = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=shezhen_32_mean, std=shezhen_32_std)
    ])

    # 基础 transform（用于初步统一尺寸）
    resize_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    transform_fixmatch = TransformFixMatch(mean=shezhen_32_mean, std=shezhen_32_std, image_size=image_size)


    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # 统一尺寸
        transforms.ToTensor(),
        # 其他 transform，例如 Normalize
    ])

 
    # 把 resize_transform 传进去做底层图像尺寸统一
    train_labeled_dataset = SheZhenDataset(label_root, train=True, unlabeled=False, transform=resize_transform,image_size=image_size)
    train_unlabeled_dataset = SheZhenDataset(unlabeled_root, train=True, unlabeled=True, transform=resize_transform,image_size=image_size)
    test_dataset = SheZhenDataset(label_root, train=False, transform=resize_transform,image_size=image_size)

    # 外包装（包装后的 transform 才能保证增强或 Normalize）
    train_labeled_dataset = CIFAR10SSL(train_labeled_dataset, train=True, transform=transform_labeled)
    train_unlabeled_dataset = CIFAR10SSL(train_unlabeled_dataset, train=True, transform=transform_fixmatch)
    test_dataset = CIFAR10SSL(test_dataset, train=False, transform=transform_val)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset



class TransformFixMatch(object):
    def __init__(self, mean=normal_mean, std=normal_std, image_size=image_size):
        self.weak = transforms.Compose([
                transforms.Resize((image_size, image_size)),  # ✅ 强制调整尺寸
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=image_size, padding=int(image_size*0.125), padding_mode='reflect')
            ])
        self.strong = transforms.Compose([
                transforms.Resize((image_size, image_size)),  # ✅ 强制调整尺寸
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=image_size, padding=int(image_size*0.125), padding_mode='reflect'),
                RandAugmentMC(n=2, m=10)
            ])
        self.normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])


    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='CIFAR10/100 dataset')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'],
                        help='dataset name')
    parser.add_argument('--root', type=str, default='./data',
                        help='data root directory')
    parser.add_argument('--num_labeled', type=int, default=4000,
                        help='number of labeled data')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='number of classes')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--expand_labels', action='store_true',
                        help='expand labels for unlabeled data')
    args = parser.parse_args()
    label_root="/git/datasets/fixmatch_dataset"
    unlabeled_root="/git/datasets/shezhen_original_data/shezhen_unlabel_data"
    args.dataset = "cifar10"
    DATASET_GETTERS = {'shezhen': get_shezhen9}

    dataset_getter = DATASET_GETTERS[args.dataset]
    train_labeled_dataset, train_unlabeled_dataset, test_dataset = dataset_getter(args, label_root,unlabel_root)
    # 输出train_labeled_dataset和train_unlabeled_dataset中的数据包含哪些
    for i in range(5):
        print(f"Train Labeled Dataset {i}: {train_labeled_dataset[i]}")
        print(f"Train Unlabeled Dataset {i}: {train_unlabeled_dataset[i]}")
        print(f"Test Dataset {i}: {test_dataset[i]}")
    