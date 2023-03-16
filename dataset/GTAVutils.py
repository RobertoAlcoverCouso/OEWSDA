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
Label = namedtuple('Label', [
                   'name',
                   'id',
                   'trainId',
                   'category',
                   'categoryId',
                   'hasInstances',
                   'ignoreInEval',
                   'color'])
labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        1 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        2 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        3 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        4 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        5 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        6 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        7 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        8 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        9 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,       10 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       11 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       12 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       13 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       15 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       17 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       18 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       19 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]
MSS_translate = {-1:0, 255:0, 1:1, 2:2, 3:3, 4:3,
                 5:3, 6:6, 7:7, 8:5, 9:8, 10:8, 11:9,
                 12:10, 13:10, 14:11, 15:12, 16:12,
                 17:12, 18:0, 19:0}
color2index = {}



color2index[(0,0,0)] = 0  # add an void class
for obj in labels:
    if obj.ignoreInEval:
        continue
    idx   = obj.trainId
    label = obj.name
    color = obj.color
    color2index[color] = idx

root_dir = "./../GTAV/"
images_dir = os.path.join(root_dir, "images")
gt_dir = os.path.join(root_dir, "gtCS")
labels_dir = os.path.join(root_dir, "labels")
if not os.path.exists(gt_dir):
    os.makedirs(gt_dir)
train_label_file = os.path.join(root_dir, "trainCS.csv")  # train file
csv_file = open(train_label_file, "w")
csv_file.write("img,label\n")
for name in os.listdir(images_dir):
    img_name = os.path.join(images_dir, name)
    fine_label = os.path.join(labels_dir, name)
    label_np_name = os.path.join(gt_dir, name + "CS")
    if os.path.exists(label_np_name):
        csv_file.write("{},{}\n".format(img_name, label_np_name ))
        continue
    print(fine_label)
    try:
        image = Image.open(fine_label)
    except:
        continue
    labels = np.asarray(image.convert("RGB"))
    height, width, _ = labels.shape
    label = np.zeros((height, width))
    for h in range(height):
        for w in range(width):
            color = tuple(labels[h, w])
            try:
                index = color2index[color]
                label[h, w] = index
            except:
                # no index, assign to void
                label[h, w] = 0
    label = label.astype(np.uint8)
    Image.fromarray(label, mode='L').save(label_np_name+ "CS.png")
    csv_file.write("{},{}\n".format(img_name, label_np_name+ "CS.png"))

csv_file.close()
