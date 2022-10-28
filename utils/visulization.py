import os
import os.path as osp
import numpy as np
import cv2
from .bbox import xywh2xyxy


def visulize_batch(out_path, imgs, targets, batch_idx):
    os.makedirs(out_path, exist_ok=True)
    imgs = imgs.cpu().numpy()
    targets = targets.cpu().numpy()

    batch_size = imgs.shape[0]
    for i in range(batch_size):
        img = np.ascontiguousarray(
            (imgs[i].transpose((1, 2, 0))[:, :, ::-1] * 255.).astype(np.uint8)
        )
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

        img_path = osp.join(out_path, f'batch{batch_idx}_{i}.png')
        cv2.imwrite(img_path, img)


def visualize_dataloader(dataloader, out_dir):
    for batch_idx, (imgs, targets) in enumerate(dataloader):
        visulize_batch(out_dir, imgs, targets, batch_idx)
        if batch_idx == 10:
            break