import logging
import math
import os
import csv
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from .randaugment import RandAugmentMC
from dataset.cifar import CIFAR10SSL, TransformFixMatch

logger = logging.getLogger(__name__)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)



class SheZhenDataset(Dataset):
    def __init__(self, root, train=True, unlabeled=False, transform=None):
        self.root = root
        self.train = train
        self.unlabeled = unlabeled 
        self.transform = transform

        if train:
            if unlabeled:
                self.data_dir = os.path.join(root, 'train', 'unlabeled')
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
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            labels = []

        if self.transform:
            image = self.transform(image)

        label_vector = self._encode_labels(labels)
        return image, label_vector

    def _encode_labels(self, labels):
        all_classes = ['气虚质', '阳虚质', '阴虚质', '痰湿质', '湿热质', '血瘀质', '气郁质', '特禀质', '平和质']
        label_vector = [1 if cls in labels else 0 for cls in all_classes]
        return torch.tensor(label_vector, dtype=torch.float32)  




def get_shezhen9(args, root):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])

    root="/git/datasets/fixmatch_dataset"

    train_labeled_dataset = SheZhenDataset(root, train=True, unlabeled=False)
    train_unlabeled_dataset = SheZhenDataset(root, train=True, unlabeled=True)
    test_dataset = SheZhenDataset(root, train=False)

    """
    train_labeled_idxs = list(range(len(train_labeled_dataset)))
    train_unlabeled_idxs = list(range(len(train_unlabeled_dataset)))"""

    train_labeled_dataset = CIFAR10SSL(
        train_labeled_dataset, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR10SSL(
        train_unlabeled_dataset, train=True,
        transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std))

    test_dataset = CIFAR10SSL(
        test_dataset, train=False, transform=transform_val)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset




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
    root="/git/datasets/fixmatch_dataset"
    args.dataset = "cifar10"
    DATASET_GETTERS = {'shezhen': get_shezhen9}

    dataset_getter = DATASET_GETTERS[args.dataset]
    train_labeled_dataset, train_unlabeled_dataset, test_dataset = dataset_getter(args, args.root)
    # 输出train_labeled_dataset和train_unlabeled_dataset中的数据包含哪些
    for i in range(5):
        print(f"Train Labeled Dataset {i}: {train_labeled_dataset[i]}")
        #print(f"Train Unlabeled Dataset {i}: {train_unlabeled_dataset[i]}")
        #print(f"Test Dataset {i}: {test_dataset[i]}")
    # 