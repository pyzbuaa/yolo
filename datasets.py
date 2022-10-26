import cv2
import warnings
import os.path as osp
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torch


class YOLODataset(Dataset):
    def __init__(self, list_path, transforms=None, with_gt=True):
        super(YOLODataset, self).__init__()
        with open(list_path, 'r') as f:
            self.img_paths = f.readlines()

        if with_gt:
            self.label_paths = []
            for img_path in self.img_paths:
                img_dir = osp.dirname(img_path)
                label_dir = "labels".join(img_dir.rsplit("JPEGImages", 1))
                assert label_dir != img_dir, \
                    f"Image path must contain a folder named 'images'! \n'{img_dir}'"
                label_path = osp.join(label_dir, osp.basename(img_path))
                label_path = osp.splitext(label_path)[0] + '.txt'
                self.label_paths.append(label_path)

        self.max_objects = 100
        self.transforms = transforms
        self.with_gt = with_gt

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index].rstrip()
        img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)

        if self.with_gt:
            label_path = self.label_paths[index].rstrip()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                labels = np.loadtxt(label_path).reshape(-1, 5)
        else:
            labels = None

        if self.transforms:
            img, labels = self.transforms(img, labels)

        return img, labels

    @staticmethod
    def collate_fn(batch):
        imgs, targets = zip(*batch)
        imgs = torch.stack(imgs, dim=0)

        batch_targets = []
        for i, target in enumerate(targets):
            target_i = torch.zeros((target.shape[0], 6))
            target_i[:, 0] = i
            target_i[:, 1:] = target
            batch_targets.append(target_i)
        batch_targets = torch.cat(batch_targets, dim=0)

        return imgs, batch_targets

