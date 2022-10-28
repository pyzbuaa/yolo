num_classes = 20

data = dict(
    train_txt = '/home/pyz/data/voc/train.txt',
    img_size=416,
    batch_size=4,
    subdivision=1
)

model = dict(
    anchors=[[(10, 13), (16, 30), (33, 23)],
            [(30, 61), (62, 45), (59, 119)],
            [(116, 90), (156, 198), (373, 326)]]
)

optimizer = dict(
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0005
)

lr_scheduler = dict(
    epoch_based=dict(milestones=[16, 19], gamma=0.1)
    # iter_based=dict(burn_in=1000)
)

work_dir = './work_dir/yolov3_voc/'
pretrain = None
epochs = 20
