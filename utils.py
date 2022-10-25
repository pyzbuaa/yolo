import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator


def xywh2xyxy(xywh):
    xyxy = np.zeros_like(xywh)
    xyxy[..., 0] = xywh[..., 0] - xywh[..., 2] / 2
    xyxy[..., 1] = xywh[..., 1] - xywh[..., 3] / 2
    xyxy[..., 2] = xywh[..., 0] + xywh[..., 2] / 2
    xyxy[..., 3] = xywh[..., 1] + xywh[..., 3] / 2
    return xyxy


def vis_batch(imgs, targets, prefix):
    imgs = imgs.cpu().numpy()
    targets = targets.cpu().numpy()

    unique_labels = targets[:, 0].unique()
    n_classes_in_batch = len(unique_labels)

    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, n_classes_in_batch)]
    bbox_colors = random.sample(colors, n_classes_in_batch)

    batch_size = imgs.shape[0]
    for i in range(batch_size):
        img = (imgs[i].transpose((1, 2, 0)) * 255.).astype(np.uint8)
        target = targets[targets[:, 0] == i][:, 1:]


if __name__ == '__main__':
    xywh = np.array(
        [[1, 2, 3, 4],
         [5, 6, 7, 8]]
    )

    print(xywh2xyxy(xywh))