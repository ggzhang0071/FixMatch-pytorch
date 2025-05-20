import os
from PIL import Image
import numpy as np
from tqdm import tqdm

def get_all_image_paths(root_dir, exts={".jpg", ".jpeg", ".png"}):
    image_paths = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in exts:
                image_paths.append(os.path.join(root, file))
    return image_paths

def compute_mean_std(image_paths, resize=(224, 224)):
    means = []
    stds = []
    for img_path in tqdm(image_paths, desc="Computing mean and std"):
        try:
            img = Image.open(img_path).convert('RGB').resize(resize)
            img_array = np.array(img) / 255.0  # (H, W, C)
            means.append(np.mean(img_array, axis=(0, 1)))
            stds.append(np.std(img_array, axis=(0, 1)))
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            os.remove(img_path)  # Remove the corrupted file
            continue
    mean = np.mean(means, axis=0)
    std = np.mean(stds, axis=0)
    return mean, std

if __name__ == "__main__":
    data_dir = "/git/datasets/shezhen_original_data"  # 替换为你的数据集路径
    image_paths = get_all_image_paths(data_dir)
    print(f"Found {len(image_paths)} images.")
    image_size = 224
    
    if not image_paths:
        print("No images found. Please check the path.")
    else:
        mean, std = compute_mean_std(image_paths, resize=(image_size, image_size))
        print("Shezhen image size {} mean:{}".format(image_size,mean))
        print("Shezhen image size {} std:{}".format(image_size,std))
