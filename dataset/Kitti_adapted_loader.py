# -*- coding: utf-8 -*-

from __future__ import print_function

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import scipy.misc
import random
import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import utils
from torchvision import transforms


root_dir   = "./../Kitti/training/"
train_file = os.path.join(root_dir, "train.csv")

means     = np.array([103.939, 116.779, 123.68]) / 255. # mean of three channels in the order of BGR




class KittiDataset(Dataset):

    def __init__(self, csv_file, phase, n_class=13, crop=False, flip_rate=0.):
        self.data      = pd.read_csv(csv_file)
        self.preprocess_img = transforms.Compose([
                            transforms.Resize(256),
                            transforms.RandomCrop(224),
                            transforms.RandomVerticalFlip(p=0.5),
                            #transforms.ToTensor(),
                            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ])
        self.preprocess_lab = transforms.Compose([
                            transforms.Resize(256),
                            transforms.ToTensor()
                        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name   = self.data.iloc[idx, 0]
        img = Image.open(img_name)
        input_tensor = self.preprocess_img(img)
        label_name = self.data.iloc[idx, 1]
        #label      = self.preprocess_lab(np.load(label_name))
        """
        # create one-hot encoding
        h, w = label.size()
        target = torch.zeros(self.n_class, h, w)
        for c in range(self.n_class):
            target[c][label == c] = 1
        """
        sample = {'X': img}

        return sample


def show_batch(batch):
    img_batch = batch['X']
    img_batch[:,0,...].add_(means[0])
    img_batch[:,1,...].add_(means[1])
    img_batch[:,2,...].add_(means[2])
    batch_size = len(img_batch)

    grid = utils.make_grid(img_batch)
    plt.imshow(grid.numpy()[::-1].transpose((1, 2, 0)))

    plt.title('Batch from dataloader')


if __name__ == "__main__":
    train_data = KittiDataset(csv_file=train_file, phase='train')

    # show a batch
    batch_size = 4
    for i in range(batch_size):
        sample = train_data[i]
        print(i, np.mean(sample['X']))

    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=4)

    for i, batch in enumerate(dataloader):
        print(i, batch['X'].size)
   
