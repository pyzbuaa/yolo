import torch

from .bbox import bbox_overlaps


class YOLOAssigner(object):
    def __init__(self, pos_thresh, neg_thresh, low_quality_iou=0.1):
        """
        
        """
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh
        self.low_quality_iou = low_quality_iou

    def assign(self, anchors, anchor_masks, gt_bboxes):
        """
        Matching on an image.

        Args:
            anchors (torch.Tensor): anchors on one image.
                shape=(N, 4), [xmin, ymin, xmax, ymax] on image scale.
            anchor_masks (torch.Tensor): anchor masks, shape=(N, )
            gt_bboxes (torch.Tensor): gt bboxes on one image.
                shape=(N, 4+), [xmin, ymin, xmax, ymax] on image scale.

        assigned_gt_inds (torch.Tensor): shape=(N, ),
            -1: ignore, 0: negtive, 1+: postive, gt id.
        """
        num_anchors = anchors.shape[0]
        num_gts = gt_bboxes.shape[0]

        # shape=(num_gts, num_anchors)
        overlaps = bbox_overlaps(gt_bboxes, anchors)

        # step 1, assgned -1 (ignore)
        assigned_gt_inds = overlaps.new_full((num_anchors, ),
                                              -1,
                                              dtype=torch.long)
        if num_gts == 0:
            assigned_gt_inds[:] = 0
            return assigned_gt_inds

        # step2, assign negative
        # for each anchor, the max iou of all gts
        # for each anchor, the best gt idx
        # shape = (num_anchors, )
        max_overlaps, argmax_overlaps = overlaps.max(dim=0)
        neg_inds = (max_overlaps >= 0) & (max_overlaps <= self.neg_thresh)
        assigned_gt_inds[neg_inds] = 0

        # step3, assign positive
        pos_inds = (max_overlaps > self.pos_thresh) & anchor_masks.type(torch.bool)
        assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1

        # step4, for each gt, assign itself to the nearest anchor.
        gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)
        for i in range(num_gts):
            if gt_max_overlaps[i] > self.low_quality_iou:
                if anchor_masks[gt_argmax_overlaps[i]]:
                    assigned_gt_inds[gt_argmax_overlaps[i]] = i + 1

        return assigned_gt_inds
