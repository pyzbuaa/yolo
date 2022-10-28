import numpy as np
import torch


def xywh2xyxy(xywh):
    xyxy = np.zeros_like(xywh)
    xyxy[..., 0] = xywh[..., 0] - xywh[..., 2] / 2
    xyxy[..., 1] = xywh[..., 1] - xywh[..., 3] / 2
    xyxy[..., 2] = xywh[..., 0] + xywh[..., 2] / 2
    xyxy[..., 3] = xywh[..., 1] + xywh[..., 3] / 2
    return xyxy


def bbox_wh_iou(wh1, wh2):
    """
    TODO (reimp)
    """
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


if __name__ == '__main__':
    wh1 = torch.tensor([116/32, 90/32])
    wh2 = torch.tensor([[2, 2], [2.5, 2.5]])
    ious = bbox_wh_iou(wh1, wh2)
    print(ious)