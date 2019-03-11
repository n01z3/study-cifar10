import os
from collections import OrderedDict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from albumentations import Normalize, Compose, \
    RandomCrop, ShiftScaleRotate, RandomGamma, RandomBrightnessContrast, \
    CenterCrop
from albumentations.torch import ToTensor
from torch.utils import data

from n01_config import get_params, get_paths

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
klass2idx = dict([(klass, n) for n, klass in enumerate(LABELS)])
idx2klass = dict(enumerate(LABELS))


def post_transform():
    return Compose([
        Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD),
        ToTensor()])


def train_transform(size=28):
    return Compose([
        RandomCrop(size, size),
        ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=(0.8, 1.2),
            rotate_limit=10,
            border_mode=cv2.BORDER_REPLICATE,
            p=0.3),
        RandomGamma(gamma_limit=(95, 105), p=0.3),
        RandomBrightnessContrast(0.1, p=0.3),
        post_transform()
    ])


def valid_transform(size=28):
    return Compose([
        CenterCrop(size, size),
        post_transform()
    ])


class CifarDataset:
    def __init__(self, mode, data_params, path, df, transform):
        assert mode, ['train', 'valid', 'test']
        self._data_params = data_params
        self._path = path
        self._transform = transform

        if mode == 'train':
            df = df[df['fold_id'] != data_params['fold_id']]
        elif mode == 'valid':
            df = df[df['fold_id'] == data_params['fold_id']]

        if mode == 'test':
            self._image_dir = path['test_dir']
        else:
            self._image_dir = path['train_dir']

        df.reset_index(inplace=True, drop=True)
        self._df = df

    def __getitem__(self, index):
        if isinstance(index, torch.Tensor):
            index = index.item()

        batch = OrderedDict()
        sample_id = self._df.loc[index, 'id']
        batch['id'] = sample_id

        klass = self._df.loc[index, 'label']
        batch['y'] = klass2idx.get(klass)

        filename = f'{self._df.loc[index, "id"]}.png'
        image_path = os.path.join(self._path['path'], self._image_dir, filename)

        image = cv2.imread(image_path)[:, :, ::-1]
        image = self._transform(image=image)['image']
        batch['image'] = image

        return batch

    def __len__(self):
        return self._df.shape[0]


def check_iter():
    batch_size = 8
    paths = get_paths()['dataset']
    data_params = get_params()['data_params']

    # df = pd.read_csv('tables/folds_n01.csv')
    # df = pd.read_csv(os.path.join(paths['path'], paths['sample']))
    df = pd.read_csv('subm/subm1.csv')

    print(df.head())

    train_dataset = CifarDataset('test', data_params, paths, df, valid_transform())
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size,
                                   shuffle=False, num_workers=12, drop_last=False)

    for batch in train_loader:
        img = batch['image']
        y = batch['y'].numpy()
        img = np.transpose(img.numpy(), (0, 2, 3, 1))

        plt.figure(figsize=(25, 35))
        for i in range(batch_size):
            plt.subplot(2, batch_size // 2, i + 1)
            plt.title(idx2klass.get(y[i]))
            shw = IMAGENET_STD * img[i] + IMAGENET_MEAN
            plt.imshow(shw)

        plt.show()


def get_loaders(batch_size=8):
    paths = get_paths()['dataset']
    data_params = get_params()['data_params']

    df = pd.read_csv('tables/folds_n01.csv')

    train_dataset = CifarDataset('train', data_params, paths, df, train_transform())
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size,
                                   shuffle=True, num_workers=32, drop_last=False)

    valid_dataset = CifarDataset('valid', data_params, paths, df, valid_transform())
    valid_loader = data.DataLoader(valid_dataset, batch_size=batch_size,
                                   shuffle=False, num_workers=32, drop_last=False)

    test_df = pd.read_csv(os.path.join(paths['path'], paths['sample']))
    test_dataset = CifarDataset('test', data_params, paths, test_df, valid_transform())
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size,
                                  shuffle=False, num_workers=32, drop_last=False)

    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    check_iter()
