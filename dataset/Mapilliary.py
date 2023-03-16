#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 17:40:46 2021

@author: e321075
"""

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import pandas as pd
import cv2
import scipy.misc
from skimage.transform import resize
import os
from collections import namedtuple
import re
import json

#############################
	# global variables #
#############################
root_dir          = "./../Mapilliary/set/"
training_set      = "validation/"
training_dir      = os.path.join(root_dir, training_set)
img_dir           = os.path.join(training_dir, "images/")
gt_dir            = os.path.join(training_dir, "labels/")
f = open(os.path.join(root_dir, "config.json"))
labels        = json.load(f)['labels']
class_label_dict  = {}
for label_id, label in enumerate(labels):
    if "pedestrian-area" in label["name"]:
        class_label_dict[label_id] = 2
    elif "sidewalk" in label["name"]:
        class_label_dict[label_id] = 2
    elif "barrier" in label["name"]:
        class_label_dict[label_id] = 3
    elif "flat" in label["name"]:
        class_label_dict[label_id] = 1
    elif "building" in label["name"]:
        class_label_dict[label_id] = 3
    elif "human" in label["name"]:
        class_label_dict[label_id] = 10
    elif "sky" in label["name"]:
        class_label_dict[label_id] = 9
    elif "nature" in label["name"]:
        class_label_dict[label_id] = 8
    elif "pole" in label["name"]:
        class_label_dict[label_id] = 6
    elif "light" in label["name"]:
        class_label_dict[label_id] = 7
    elif "car" in label["name"]:
        class_label_dict[label_id] = 11
    elif "bus" in label["name"]:
        class_label_dict[label_id] = 12
    elif "sign" in label["name"]:
        class_label_dict[label_id] = 5
    
np_labels         = os.path.join(root_dir, "np_labels_val/")
train_label_file  = os.path.join(root_dir, "val.csv") # train file
csv_file = open(train_label_file, "w")
#csv_file.write("img,label\n")
for idx, img in enumerate(os.listdir(img_dir)):
    img_name = os.path.join(img_dir, img)
    label_name = os.path.join(gt_dir, img[:-3]+"png")
    gt_name = os.path.join(np_labels, img[:-4])
    if os.path.exists(gt_name):
        print("Skip %s" % (gt_name))
        continue
    image = Image.open(label_name)
    labels = np.asarray(image)
    height, weight = labels.shape
    label = np.zeros((height,weight))
    #print(labels.shape)
    for h in range(height):
        for w in range(weight):
            try:
                label[h,w] = class_label_dict[labels[h,w]]
            except:
                label[h,w] = 0
    
    
    np.savez_compressed(gt_name, label)
    csv_file.write("{},{}\n".format(img_name, os.path.join(np_labels, img[:-4]) + ".npz"))
            
csv_file.close()
