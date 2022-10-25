from datasets import YOLODataset
from transforms import Compose, Resize, Pad, ToTensor

import torchvision.transforms as transforms


def create_dataloader(list_path, img_size, multiscale, with_gt=True):
    data_transforms = Compose([
        Resize(416),
        Pad(416),
        ToTensor()
    ])
    
    dataset = YOLODataset(
        list_path,
        transforms=data_transforms,
        with_gt=with_gt
    )

    for img, label in dataset:
        print(img.shape)
        print(label.shape)


if __name__ == '__main__':
    train_txt = '/home/pyz/data/voc/train.txt'
    img_size = 416
    multiscale = False
    create_dataloader(train_txt, img_size, multiscale)