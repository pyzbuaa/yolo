import imp
import torch
from utils.bbox import bbox_wh_iou


def build_target(targets, preds, anchors):
    """
    Args:
        targets (Tensor): shape=(N, 6), [batch_id, class_id, x, y, w, h],
            normalized by image size.
        preds (Tensor): shape=(B, num_anchors, H, W, 5+C)
        anchors (Tensor): shape=(3, 2), anchors of current level.

    Returns:

    """

    B = preds.shape[0]
    n_anchros = preds.shape[1]
    n_classes = preds.shape[-1] - 5
    H = preds.shape[2]
    W = preds.shape[3]

    obj_mask = torch.Tensor(B, n_anchros, H, W).fill_(0)
    noobj_mask = torch.Tensor(B, n_anchros, H, W).fill_(1)

    scales = torch.ones(6)
    scales[2:6] = torch.tensor([W, H, W, H])
    targets *= scales # (N, 6), in feature map

    bboxes_xy = targets[[2, 3]] # (N, 2)
    bboxes_wh = targets[[4, 5]] # (N, 2)
