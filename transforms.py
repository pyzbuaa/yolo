import cv2
import numpy as np
import torch


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target


class ToRelativeCoord(object):
    def __call__(self, img, target):
        h, w, _ = img.shape
        target[:, [1, 3]] /= w
        target[:, [2, 4]] /= h
        return img, target


class ToAbsCoord(object):
    def __call__(self, img, target):
        h, w, _ = img.shape
        target[:, [1, 3]] *= w
        target[:, [2, 4]] *= h
        return img, target


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        ori_h, ori_w, _ = img.shape
        scale = self.size / max(ori_h, ori_w)
        dst_h = int(scale * ori_h)
        dst_w = int(scale * ori_w)
        img = cv2.resize(img, (dst_w, dst_h))
        return img, target


class Pad(object):
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, img, target):
        dst_h, dst_w = self.size
        ori_h, ori_w, _ = img.shape
        pad_h = dst_h - ori_h
        pad_w = dst_w - ori_w

        tl_x = pad_w // 2
        tl_y = pad_h // 2

        padded_img = np.zeros((dst_h, dst_w, 3), dtype=img.dtype)
        padded_img[tl_y : tl_y + ori_h, tl_x : tl_x + ori_w, :] = img

        # to abs coord
        target[:, [1, 3]] *= ori_w
        target[:, [2, 4]] *= ori_h

        target[:, 1] += tl_x
        target[:, 2] += tl_y

        # to relative coord
        target[:, [1, 3]] /= dst_w
        target[:, [2, 4]] /= dst_h

        return padded_img, target


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, img, target):
        default_float_dtype = torch.get_default_dtype()
        img = torch.from_numpy(img.transpose((2, 0, 1))).contiguous()
        if isinstance(img, torch.ByteTensor):
            img = img.to(dtype=default_float_dtype).div(255)

        target = torch.from_numpy(target)

        return img, target
