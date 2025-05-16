from torchvision import datasets, transforms

def get_shezhen_data(args):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dir = args.root_dir + "/train"
    test_dir = args.root_dir + "/test"
    

    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform_train)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform_test)

    # FixMatch 还需要返回无标签数据集，暂时先用空列表或复制train_dataset
    train_unlabeled_dataset = train_dataset

    return train_dataset, train_unlabeled_dataset, test_dataset


if __name__ == "__main__":
    train_dir = "/git/fixmatch_dataset/train"
    test_dir = "/git/fixmatch_dataset/test"
    batch_size = 64
    num_workers = 4

    train_loader, test_loader = get_fixmatch_dataloaders(train_dir, test_dir, batch_size, num_workers)

    # 测试数据加载器
    for images, labels in train_loader:
        print(images.shape, labels.shape)
        break
