from datasets import YOLODataset
from datasets import Compose, Resize, Pad, ToTensor
from utils.utils import worker_seed_set
from utils import Config
from utils.visulization import visualize_dataloader
from models.model import YOLO, build_model

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

import argparse
from tqdm import tqdm
import os.path as osp
from terminaltables import AsciiTable


def build_train_dataloader(list_path, img_size, batch_size, with_gt=True):
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


def build_optimizer(model, **kwargs):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, **kwargs)
    return optimizer


def build_scheduler(optimizer, **kwargs):
    return optim.lr_scheduler.MultiStepLR(optimizer, **kwargs)


def train(model, dataloader, optimizer, scheduler, subdivisions, epochs, device):
    for epoch in range(1, epochs + 1):
        model.train()

        for batch_idx, (imgs, targets) in enumerate(dataloader):
            total_batch = len(dataloader) * epoch + batch_idx

            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device)
            outputs = model(imgs)

            print(outputs[0].shape)

            # loss
            # loss.backward()

            # real batch
            if total_batch % subdivisions == 0:
                optimizer.step()
                optimizer.zero_grad()

        scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train YOLO')
    parser.add_argument('config', type=str, help='config file')
    args = parser.parse_args()

    cfg = Config.frompy(args.config)
    data_cfg = cfg.data
    model_cfg = cfg.model

    batch_size = data_cfg.batch_size
    subdivision = data_cfg.subdivision
    mini_batch_size = batch_size // subdivision
    train_loader = build_train_dataloader(data_cfg.train_txt,
                                          data_cfg.img_size,
                                          mini_batch_size)
    # check dataset before training.
    data_samples_dir = osp.join(cfg.work_dir, 'data_samples')
    visualize_dataloader(train_loader, data_samples_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_cfg.anchors,
                        cfg.num_classes,
                        device,
                        checkpoint=cfg.pretrain)
    optimizer = build_optimizer(model, **cfg.optimizer)
    scheduler = build_scheduler(optimizer, **cfg.lr_scheduler.epoch_based)

    train(model, train_loader, optimizer, scheduler, subdivision, cfg.epochs, device)