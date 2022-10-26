import cv2
import os
import os.path as osp
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

import torch


def worker_seed_set(worker_id):
    # See for details of numpy:
    # https://github.com/pytorch/pytorch/issues/5059#issuecomment-817392562
    # See for details of random:
    # https://pytorch.org/docs/stable/notes/randomness.html#dataloader

    # NumPy
    uint64_seed = torch.initial_seed()
    ss = np.random.SeedSequence([uint64_seed])
    np.random.seed(ss.generate_state(4))

    # random
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)


def xywh2xyxy(xywh):
    xyxy = np.zeros_like(xywh)
    xyxy[..., 0] = xywh[..., 0] - xywh[..., 2] / 2
    xyxy[..., 1] = xywh[..., 1] - xywh[..., 3] / 2
    xyxy[..., 2] = xywh[..., 0] + xywh[..., 2] / 2
    xyxy[..., 3] = xywh[..., 1] + xywh[..., 3] / 2
    return xyxy


def vis_batch(out_path, imgs, targets, num_classes, prefix):
    os.makedirs(out_path, exist_ok=True)
    imgs = imgs.cpu().numpy()
    targets = targets.cpu().numpy()

    batch_size = imgs.shape[0]
    for i in range(batch_size):
        img = np.ascontiguousarray((imgs[i].transpose((1, 2, 0))[:, :, ::-1] * 255.).astype(np.uint8))
        target = targets[targets[:, 0] == i][:, 1:]

        img_h, img_w, _ = img.shape
        bboxes = target[:, 1:]
        bboxes[:, [0, 2]] *= img_w
        bboxes[:, [1, 3]] *= img_h
        bboxes = xywh2xyxy(bboxes)
        for bbox in bboxes:
            cv2.rectangle(img,
                         (int(bbox[0]), int(bbox[1])),
                         (int(bbox[2]), int(bbox[3])),
                         (0, 255, 0),
                         thickness=1)

        img_path = osp.join(out_path, f'batch{prefix}_{i}.png')
        cv2.imwrite(img_path, img)


if __name__ == '__main__':
    xywh = np.array(
        [[1, 2, 3, 4],
         [5, 6, 7, 8]]
    )

    print(xywh2xyxy(xywh))