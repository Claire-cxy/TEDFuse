#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os.path as osp
import os
from PIL import Image
import numpy as np
import json

from transform import *


class CityScapes(Dataset):
    def __init__(
        self,
        rootpth,
        cropsize=(640, 480),
        # mode='train',
        Method='newFusion'
    ):
        super(CityScapes, self).__init__()
        # assert mode in ('train', 'val', 'test')
        # self.mode = mode
        self.ignore_lb = 255

        with open('./cityscapes_info.json', 'r') as fr:
            labels_info = json.load(fr)
        self.lb_map = {el['id']: el['trainId'] for el in labels_info}

        ## parse img directory
        self.imgs = {}
        imgnames = []
        # impth = osp.join(rootpth, Method, mode)
        impth = osp.join(rootpth, Method)
        print('impth' + impth)
        # folders = os.listdir(impth)
        # print('folders' + folders)
        # 遍历一个文件夹中的所有子文件夹，并将子文件夹中的所有图片文件名和路径存储在一个字典中
        for fd in os.listdir(impth):
            fdpth = osp.join(impth, fd)
            if os.path.isfile(fdpth):
                # im_names = os.listdir(fdpth)
                names = fd.replace('.png', '')
                imgnames.append(names)
                obj = dict({names: fdpth})
                self.imgs.update(obj)

        ## parse gt directory
        self.labels = {}
        gtnames = []
        gtpth = osp.join(rootpth, 'Segmentation_labels')
        folders = os.listdir(gtpth)
        for fd in folders:
            fdpth = osp.join(gtpth, fd)
            if os.path.isfile(fdpth):
                # im_names = os.listdir(fdpth)
                names = fd.replace('.png', '')
                gtnames.append(names)
                obj = dict({names: fdpth})
                self.labels.update(obj)

        self.imnames = imgnames
        self.len = len(self.imnames)
        assert set(imgnames) == set(gtnames)
        assert set(self.imnames) == set(self.imgs.keys())
        assert set(self.imnames) == set(self.labels.keys())

        ## pre-processing
        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize(
                #     (0.2254, 0.2585, 0.2225), (0.0521, 0.0529, 0.0546)
                # ),
            ]
        )
        self.trans_train = Compose(
            [
                ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                HorizontalFlip(),
                RandomScale((0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
                RandomCrop(cropsize),
            ]
        )

    def __getitem__(self, idx):
        fn = self.imnames[idx]
        impth = self.imgs[fn]
        if os.path.isdir(impth):
            return

        lbpth = self.labels[fn]
        img = Image.open(impth)
        label = Image.open(lbpth)
        # if self.mode == 'train':
        im_lb = dict(im=img, lb=label)
        im_lb = self.trans_train(im_lb)
        img, label = im_lb['im'], im_lb['lb']
        # test_image = np.array(img)
        img = self.to_tensor(img)
        label = np.array(label).astype(np.int64)[np.newaxis, :]
        return img, label, fn

    def __len__(self):

        return self.len

    def convert_labels(self, label):
        for k, v in self.lb_map.items():
            label[label == k] = v
        return label


if __name__ == "__main__":
    from tqdm import tqdm

    ds = CityScapes(rootpth='./datasets/MSRS/')
    uni = []
    for im, lb ,_ in tqdm(ds):
        lb_uni = np.unique(lb).tolist()
        uni.extend(lb_uni)
    print(uni)
    print(set(uni))

