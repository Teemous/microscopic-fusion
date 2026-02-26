import os
import glob
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
import cv2 as cv

class Microscopy_dataset(Dataset):
    def __init__(self,root_dir):
        self.images1 = glob.glob(os.path.join(root_dir, '*-1.jpg'))
        self.images1.sort()
        self.images2 = glob.glob(os.path.join(root_dir, '*-2.jpg'))
        self.images2.sort()
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.images1)

    def __getitem__(self, idx):
        img1 = cv.imread(self.images1[idx])
        img2 = cv.imread(self.images2[idx])
        y1, cr, cb = cv.split(cv.cvtColor(img1, cv.COLOR_BGR2YCrCb))
        y2, _, _ = cv.split(cv.cvtColor(img2, cv.COLOR_BGR2YCrCb))

        y1 = self.transform(y1)
        y2 = self.transform(y2)

        return y1, y2, cr, cb, os.path.basename(self.images1[idx]).split('.')[0][:-2]

class Microscopy_dataset_MFI(Dataset):
    def __init__(self, root_dir, crop_divisor=32):
        super().__init__()

        self.root_dir = root_dir
        self.crop_divisor = crop_divisor

        self.dir_gt = os.path.join(root_dir, 'Image_fusion_dataset/full_clear')
        self.dir_s1 = os.path.join(root_dir, 'Image_fusion_dataset/source_1')
        self.dir_s2 = os.path.join(root_dir, 'Image_fusion_dataset/source_2')

        self.gt_list = sorted(glob.glob(os.path.join(self.dir_gt, '*.jpg')))
        assert len(self.gt_list) > 0, "full_clear 文件夹为空！"

        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.gt_list)

    def _crop_to_divisor(self, img, divisor):
        """
        img: numpy array [H, W]
        return: cropped_img, (H_crop, W_crop)
        """
        H, W = img.shape
        H_crop = (H // divisor) * divisor
        W_crop = (W // divisor) * divisor

        assert H_crop > 0 and W_crop > 0, \
            f"Image too small for divisor={divisor}, got {H}x{W}"

        return img[:H_crop, :W_crop], (H_crop, W_crop)

    def __getitem__(self, idx):
        #文件名
        gt_path = self.gt_list[idx]
        img_name = os.path.basename(gt_path)
        name_no_ext = os.path.splitext(img_name)[0]

        s1_path = os.path.join(self.dir_s1, img_name)
        s2_path = os.path.join(self.dir_s2, img_name)

        #读图
        img_gt = cv.imread(gt_path)
        img1 = cv.imread(s1_path)
        img2 = cv.imread(s2_path)

        assert img_gt is not None, f"GT not found: {gt_path}"
        assert img1 is not None, f"Source1 not found: {s1_path}"
        assert img2 is not None, f"Source2 not found: {s2_path}"

        #YCrCb
        y_gt, _, _ = cv.split(cv.cvtColor(img_gt, cv.COLOR_BGR2YCrCb))
        y1, cr, cb = cv.split(cv.cvtColor(img1, cv.COLOR_BGR2YCrCb))
        y2, _, _ = cv.split(cv.cvtColor(img2, cv.COLOR_BGR2YCrCb))

        y1 = self.transform(y1)     # [1, Hc, Wc]
        y2 = self.transform(y2)
        y_gt = self.transform(y_gt)

        cr = torch.from_numpy(cr).unsqueeze(0)  # uint8
        cb = torch.from_numpy(cb).unsqueeze(0)

        return y1, y2, cr, cb, y_gt, name_no_ext


if __name__ == "__main__":
    root_dir = "data/test"
    dataset = Microscopy_dataset(root_dir)

    print(f"样本数量: {len(dataset)}")
    print(f"第一对图像路径: {dataset.images1[0]}, {dataset.images2[0]}")

    y1, y2, cr, cb, name = dataset[0]
    print(f"样本名称: {name}")
    print(f"y1 形状: {y1.shape}, y2 形状: {y2.shape}")
    print(f"cr 形状: {cr.shape}, cb 形状: {cb.shape}")