import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


# 最大最小值归一化函数
def min_max_normalize(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    normalized_tensor = 2 * (tensor - min_val) / (max_val - min_val + 1e-5) - 1  # 归一化到 [-1, 1]
    return normalized_tensor, min_val.item(), max_val.item()


# 高分辨率图像归一化函数（使用对应低分辨率图像的最大最小值）
def hr_min_max_normalize(image, lr_min_val, lr_max_val):
    tensor = transforms.ToTensor()(image)
    normalized_tensor = 2 * (tensor - lr_min_val) / (lr_max_val - lr_min_val + 1e-5) - 1
    return normalized_tensor


# 反归一化函数
def min_max_denormalize(normalized_tensor, min_val, max_val):
    original_tensor = (normalized_tensor + 1) * (max_val - min_val + 1e-5) / 2 + min_val
    return original_tensor


# 数据集定义
class Dataset(Dataset):
    def __init__(self, lr_folders, hr_folder, lr_transform=None):
        self.lr_folders = lr_folders
        self.hr_folder = hr_folder
        self.lr_transform = lr_transform
        self.lr_image_paths = []
        self.hr_image_paths = []
        self._load_image_paths()

    def _load_image_paths(self):
        for root_dir in self.lr_folders:
            image_paths = sorted(glob.glob(os.path.join(root_dir, '*.tif')))
            self.lr_image_paths.append(image_paths)

        self.hr_image_paths = sorted(glob.glob(os.path.join(self.hr_folder, '*.tif')))

    def __len__(self):
        return len(self.hr_image_paths)

    def __getitem__(self, idx):
        lr_images = [Image.open(paths[idx]) for paths in self.lr_image_paths]
        hr_image = Image.open(self.hr_image_paths[idx])

        lr_min_vals = []
        lr_max_vals = []
        if self.lr_transform:
            lr_images_transformed = [self.lr_transform(image) for image in lr_images]
            lr_images, lr_min_vals, lr_max_vals = zip(*lr_images_transformed)
            lr_min_val, lr_max_val = lr_min_vals[0], lr_max_vals[0]
        else:
            lr_images = [transforms.ToTensor()(image) for image in lr_images]

        # 使用对应的低分辨率图像的最大最小值对高分辨率图像进行归一化
        hr_image = hr_min_max_normalize(hr_image, lr_min_val, lr_max_val)

        lr_images = torch.cat(lr_images, dim=0)  # Concatenate along the channel dimension to get shape [3, H, W]

        return lr_images, hr_image, lr_min_val, lr_max_val

    def get_hr_image_count(self):
        return len(self.hr_image_paths)


# 图像预处理
lr_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: min_max_normalize(x))
])
