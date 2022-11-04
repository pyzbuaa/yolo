import torch


class YOLOAnchorGenerator(object):
    def __init__(self, base_sizes, strides):
        """
        base_sizes (list[list[tuple[int, int]]]): The basic sizes
            of anchors in multiple levels. 
            The outer list indicates feature levels, and the inner list
            corresponds to anchors of the level. Each element of
            the inner list is a tuple of shape (anchor_w, anchor_h).
            e.g. [[(10, 13), (16, 30), (33, 23)],
                  [(30, 61), (62, 45), (59, 119)],
                  [(116, 90), (156, 198), (373, 326)]]
        strides (list[int]): Strides in multiple levels.
            e.g. [8, 16, 32]
        """
        self.strides = strides
        self.centers = [torch.Tensor((stride / 2, stride / 2)) 
                        for stride in strides]
        self.base_sizes = [torch.Tensor(base_size)
                           for base_size in base_sizes]
        self.multi_level_base_anchors = self.gen_base_anchors()

    def gen_base_anchors(self):
        """
        Generate multi-level base anchors.
        """
        multi_level_base_anchors = []
        for center, base_size in zip(self.centers, self.base_sizes):
            multi_level_base_anchors.append(
                self.gen_single_level_base_anchors(center, base_size)
            )
        return multi_level_base_anchors

    def gen_single_level_base_anchors(self, center, base_sizes):
        """
        Generate base anchors in single level feature map.

        Args:
            center (torch.Tensor): center of top left cell.
                shape=(2, ), e.g. [4., 4.]
            base_sizes (torch.Tensor): base_size in single level.
                shape=(n_anchors, 2), e.g. [[anchor_w, anchor_h], ...]

        Returns:
            torch.Tensor: base_anchors in top left cell of feat map.
                shape=(n_anchors, 4). (xmin, ymin, xmax, ymax)
        """
        center = center.unsqueeze(dim=0)
        xmin = center[:, 0] - base_sizes[:, 0] * 0.5
        ymin = center[:, 1] - base_sizes[:, 1] * 0.5
        xmax = center[:, 0] + base_sizes[:, 0] * 0.5
        ymax = center[:, 1] + base_sizes[:, 1] * 0.5
        single_level_base_anchors = torch.stack([xmin, ymin, xmax, ymax], dim=1)

        return single_level_base_anchors

    def meshgrid_anchors(self, feat_sizes):
        """
        Args:
            feat_sizes (list[tuple[int, int]]): The sizes of feature maps
                in multiple levels.
                Each element of the list is a tuple of shape (feat_h, feat_w).
                e.g. [(76, 76), (38, 38), (19, 19)]

        Returns:
            list[Tensor]
        """
        multilevel_anchors = []
        for level_id, feat_size in enumerate(feat_sizes):
            single_level_anchors = self.single_level_grid_anchors(
                self.multi_level_base_anchors[level_id],
                feat_size,
                self.strides[level_id]
            )
            multilevel_anchors.append(single_level_anchors)
        return multilevel_anchors
        
    def single_level_grid_anchors(self, base_anchor, feat_size, stride):
        """
        Generate grid anchors in single level feature map.

        Args:
            base_anchor (torch.Tensor): base anchors in single level featmap.
                shape=(n, 4), vstack of [xmin, ymin, xmax, ymax].
            feat_size (tuple[int, int]): Size of the feature maps,
                (feat_h, feat_w).
            stride (int): stride of current feature map.

        Return:
            torch.Tensor: Anchors in the overall feature maps.
                shape=(N, 4), N = 3 * feat_h * feat_w
        """
        feat_h, feat_w = feat_size
        # img scale
        offset_x = torch.arange(feat_w) * stride
        offset_y = torch.arange(feat_h) * stride
        offset_xx = offset_x.repeat(offset_y.shape[0])
        offset_yy = offset_y.view(-1, 1).repeat(1, offset_x.shape[0]).view(-1)
        offsets = torch.stack(
            [offset_xx, offset_yy, offset_xx, offset_yy], dim=1) # (feat_h * feat_w, 4)
        grid_anchors = base_anchor[None, ...] + offsets[:, None, :]
        grid_anchors = grid_anchors.view(-1, 4)

        return grid_anchors


if __name__ == '__main__':
    base_sizes = [[(10, 13), (16, 30), (33, 23)],
                  [(30, 61), (62, 45), (59, 119)],
                  [(116, 90), (156, 198), (373, 326)]]
    strides = [8, 16, 32]
    anchor_generator = YOLOAnchorGenerator(base_sizes, strides)
    anchors = anchor_generator.meshgrid_anchors([(76, 76), (38, 38), (19, 19)])
    print(anchors[0].shape)