#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 17:00:22 2021

@author: e321075
"""


from matplotlib import pyplot as plt
import os
import pandas as pd
from PIL import Image

from torchvision import datasets, transforms, models
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import scipy.io as sio
import random
import sys
import argparse
import os
import time
from os.path import join
import csv
from dataset.Cityscapes_utils import idx2color


means=np.array([86.5628,86.6691,86.7348])
means_K = np.array([0.485, 0.456, 0.406])
std=[0.229, 0.224, 0.225]
std_K = [1.33, 1.43, 1.44]

label_translation= {0:0,
                    1:1,#road
                    2:2, #sidewalk
                    3:3,#building
                    4:3, #wall
                    5:3,#billboard
                    6:6, #pole
                    7:7, #trafic light
                    8:5, #trafic sign
                    9:8, #vegetation
                    10:8,#terrain
                    11:9,#sky
                    12:10,# pedestrian
                    13:10,# rider
                    14:11, #car
                    15:12, #truck
                    16:12, #bus
                    17:12, #train
                    18:11, #moto
                    19:11, #bike
                    20:1, #roadmarks
                    21:0, #unknown
                    }
class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

class Loader(Dataset):

    def __init__(self, csv_file, phase, size=224):
        if isinstance(csv_file, dict):
            self.data = None
            for key in csv_file:
                df = pd.read_csv(key)
                if self.data is None:
                    self.data = df.sample(frac=csv_file[key], random_state=0)

                else:
                    if csv_file[key] == 1:
                        self.data = self.data.append(df)
                    else:
                        self.data = self.data.append(df.sample(frac=csv_file[key], random_state=0))
                print(len(self.data), key)
        else:
            self.data            = pd.read_csv(csv_file)
        
        self.data = self.data.dropna()
        print(len(self.data))
        self.data = self.data.reset_index(drop=True)
        self.phase           = phase
        self.train           = phase == "train"
        self.visualize       = phase == "visualize"
        self.size            = size
        self.transform_MSS = transforms.Compose([
                                    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
                                    transforms.ToTensor(),
                                    transforms.Normalize(means_K, std),
                                ])
        self.transform_test = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(means_K, std),
                                ])
        self.transform_mask  = transforms.Compose([
                                    transforms.ToTensor()
                                ])
        self.img_Denorm = DeNormalize(means, std)

    def __len__(self):
        return len(self.data)

    def transform(self, image, mask):
        # TO FILL:
        # 1st Resize image and mask to 400x400 using nearest neighbor interpolation.
        resize = TF.resize
        image  = resize(image, (512, 1024), interpolation=Image.BICUBIC)
        mask   = resize(mask, (512, 1024), interpolation=Image.NEAREST)
        # This is to ensure that crop parameters are the same for image and mask. If not, the ground-truth mask would not be aligned with its image content.
        i, j, h, w = transforms.RandomCrop(400).get_params(image, [400, 800])
        # b) Crop according to these parameters
        image = transforms.functional.crop(image, i,j,h,w)
        mask  = transforms.functional.crop(mask, i,j,h,w) 
        # TO FILL:
        # 3rd random horizontal flipping
        
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask  = TF.hflip(mask)

        return image, mask
    def visual_transforms(self, image, mask):
        resize = TF.resize
        image = resize(image, (400, 400), interpolation=Image.NEAREST)
        mask = resize(mask, (400, 400), interpolation=Image.NEAREST)
        images = []
        masks = []

        for i in range(2):
            for j in range(2):

                image_aux = transforms.functional.crop(image, 176*i, 176*j, 224, 224)
                mask_aux = transforms.functional.crop(mask, 176*i, 176*j, 224, 224)
                images.append(image_aux)
                masks.append(mask_aux)

        return images, masks


    # Default trasnformations on test data
    def test_transform(self, image, mask):

        resize = TF.resize
        image  = resize(image, (400, 800), interpolation=Image.BICUBIC)
        mask   = resize(mask, (400, 800), interpolation=Image.NEAREST)

        return image, mask

    def __getitem__(self, idx):
        img_name    = self.data.iloc[idx, 0]
        label_name  = self.data.iloc[idx, 1]
        try:
            input_image = Image.open(img_name).convert('RGB')
        except:
            print(img_name, label_name)
        try:
            mask        = Image.open(label_name)
        except:
           
        
            label       = np.load(label_name)
            if "npz" in label_name:
                label = np.array(label['arr_0'], dtype=np.uint8)
                mask = Image.fromarray(label)
            else:
                mask        = Image.fromarray(label.astype(np.uint8))
        

        # reduce mean

        if self.train:
            # TO FILL:trasnform training data
            img, mask = self.transform(input_image, mask)
            img = self.transform_MSS(img)
        elif self.visualize:
            images, masks = self.visual_transforms(input_image, mask)
            images = [self.transform_test(img) for img in images]
            masks = [255*self.transform_mask(mask) for mask in masks]
        else:
            # TO FILL:trasnform test data
            img, mask = self.test_transform(input_image, mask)
            img = self.transform_test(img)

        
        if self.transform_mask is not None:
            if not self.visualize:
                mask = 255*self.transform_mask(mask)
        if self. visualize:
            sample = {'X': images, 'Y': masks}
        else:
            sample = {'X': img, 'Y': mask.long()}

        return sample

    def label_to_RGB(self, image):
        image = image.squeeze()
        height, weight = image.shape

        rgb = np.zeros((height, weight, 3))
        for h in range(height):
            for w in range(weight):
                rgb[h,w,:] = idx2color[image[h,w]]
        return rgb.astype(np.uint8)
    def show_batch(self, batch):
        img_batch = batch['X']
        batch_size = len(img_batch)
        #print(img_batch.shape)
        #np_batch = img_batch.cpu().numpy().reshape((4,3,224,224))


        if self.visualize:
            plt.figure()

            for i in range(len(batch['X'][0])):
                image = np.zeros((400, 400, 3), dtype=np.uint8)
                for j in range(4):
                    image_np = (255 * (
                        self.img_Denorm(img_batch[j][i, ...]).data.permute(1, 2, 0).cpu().numpy())).astype(
                        np.uint8)

                    if j == 0:
                        image[:224, :224] = image_np
                    elif j==1:
                        image[:224, 176:] = image_np
                    elif j==2:
                        image[176:,:224] = image_np
                    else:
                        image[176:,176:] = image_np

                img_pil = Image.fromarray(image)
                plt.subplot(2, 4, i + 1)
                plt.imshow(img_pil, interpolation='nearest')

            mask = [np.ones((400, 400), dtype=np.uint8) for _ in range(8)]
            plt.figure()
            for j in range(4):

                labels = batch['Y'][j].cpu().numpy()

                for i in range(len(batch['X'][0])):

                    region = np.copy(labels[i])
                    if j == 0:
                        mask[i][:224, :224] = region
                    elif j==1:

                        mask[i][:224, 176:]= region
                    elif j==2:
                        mask[i][176:,:224] = region
                    else:
                        mask[i][176:,176:] = region

            for i in range(8):

                plt.subplot(2, 4, i + 1)
                plt.imshow(mask[i], interpolation='nearest')
            #plt.title('Batch from dataloader')
            plt.show()

            return
        plt.figure()
        for i in range(batch_size):
            image_np = (255*(self.img_Denorm(img_batch[i,...]).data.permute(1,2,0).cpu().numpy())).astype(np.uint8)
            
            img_pil = Image.fromarray(image_np)
            plt.subplot(2, int(batch_size/2), i+1)
            plt.imshow(img_pil, interpolation='nearest')
        plt.title('Batch from dataloader')
        plt.show()
        plt.figure()
        labels = batch['Y'].cpu().numpy()

        for i in range(batch_size):
            plt.subplot(2, int(batch_size/2), i+1)
            plt.imshow(labels[i,...].squeeze(), interpolation='nearest')
        plt.title('Labels')
        plt.figure()
        #labels = batch['Y'].cpu().numpy()

        for i in range(batch_size):
            plt.subplot(2, int(batch_size/2), i+1)
            plt.imshow(self.label_to_RGB(labels[i,...]), interpolation='nearest')
            plt.axis('off')
        plt.tight_layout()
        plt.show()



datanames_csvfiles = {"./../data/Cityscapes/train.csv": 1,
                      "./../data/GTAV/trainCS.csv":        1,
                      './../data/Synthia/train.csv':1}
if __name__ == "__main__":
    root_dir   = "./../MSS/data/"
    train_file = os.path.join(root_dir, "50CS.csv")
    train_data = Loader(csv_file="./../data/Synthia/train.csv", phase='test')

    # show a batch
    batch_size = 8


    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=1)
    for i, batch in enumerate(dataloader):
        train_data.show_batch(batch)

