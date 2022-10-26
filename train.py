from datasets import YOLODataset
from transforms import Compose, Resize, Pad, ToTensor
from utils import worker_seed_set, vis_batch
from model import YOLO

from torch.utils.data import DataLoader


def create_dataloader(list_path, img_size, batch_size, with_gt=True):
    data_transforms = Compose([
        Resize(img_size),
        Pad(img_size),
        ToTensor()
    ])
    dataset = YOLODataset(
        list_path,
        transforms=data_transforms,
        with_gt=with_gt
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
        worker_init_fn=worker_seed_set
    )

    return dataloader


if __name__ == '__main__':
    train_txt = '/home/pyz/data/voc/train.txt'
    img_size = 416
    batch_size = 8
    dataloader = create_dataloader(train_txt, img_size, batch_size=batch_size)

    out_path = 'work_dir'
    for batch_idx, (imgs, targets) in enumerate(dataloader):
