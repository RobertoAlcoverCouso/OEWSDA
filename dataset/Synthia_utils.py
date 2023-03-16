#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 17:40:46 2021

@author: e321075
"""

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import random	
import cv2
import scipy.misc
from skimage.transform import resize
import os
from collections import namedtuple
import re


#############################
	# global variables #
#############################
label_translation= {0:0,
                    1:1,#road
                    2:2, #sidewalk
                    3:3,#building
                    4:4, #wall
                    5:5,#billboard
                    6:6, #pole
                    7:7, #trafic light
                    8:8, #trafic sign
                    9:9, #vegetation
                    10:10,#terrain
                    11:11,#sky
                    12:12,# pedestrian
                    13:13,# rider
                    14:14, #car
                    15:15, #truck
                    16:16, #bus
                    17:17, #train
                    18:18, #moto
                    19:19, #bike
                    20:1, #roadmarks
                    21:0, #unknown
                    }

root_dir          = "./../Synthia/"
train_label_file  = os.path.join(root_dir, "train_np_labelsCS.csv") # train file
root_dir  = os.path.join(root_dir, "train/")
csv_file = open(train_label_file, "w")
csv_file.write("img,label\n")
for idx, name in enumerate(os.listdir(root_dir)):
    label_dir = os.path.join(root_dir, name)
    for name in os.listdir(label_dir):
        fine_label = os.path.join(label_dir, name)
        print(fine_label)
        SemSeg_dir = os.path.join(fine_label, "SemSeg")
        labels_dir = os.path.join(fine_label, "labelsCS")
        if not os.path.exists(labels_dir):
            os.makedirs(labels_dir)
        image_dir = os.path.join(fine_label, "RGB")
        for img in os.listdir(image_dir): 
            img_name = os.path.join(image_dir, img)
            label_name = os.path.join(SemSeg_dir, img)
            label_np_name = os.path.join(labels_dir, img[:-4]) + ".png"
            if os.path.exists(label_np_name):
                label = np.load(label_np_name)
                label = np.array(label['arr_0'], dtype=np.uint8)
                #print(np.max(label), np.min(label))
                #mask = Image.fromarray(label)
                label[label>19] = 0
                np.savez_compressed(label_np_name, label)
                csv_file.write("{},{}\n".format(img_name, label_np_name))
                continue
            image = Image.open(label_name)
            labels = np.asarray(image.convert("RGB"))
            labels = labels[:,:,0]
            height, weight = labels.shape
            label = np.zeros((height,weight))
            for h in range(height):
                for w in range(weight):
                    if labels[h,w] < 20:
                        label[h,w] = labels[h,w]
                    else:
                        label[h,w] = 0
            Image.fromarray(label.astype(np.uint8), mode='L').save(label_np_name)
            csv_file.write("{},{}\n".format(img_name, label_np_name))
            
csv_file.close()
