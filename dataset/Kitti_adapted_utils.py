from __future__ import print_function


from matplotlib import pyplot as plt
import numpy as np
import random	
import cv2
import scipy.misc
from PIL import Image
from skimage.transform import resize
import os
from collections import namedtuple
import re


#############################
	# global variables #
#############################
root_dir          = "./../Kitti/training/"
images_dir        = os.path.join(root_dir, "image_2")    # train images
images2_dir       = os.path.join(root_dir, "images")    # train images
semantic_dir      = os.path.join(root_dir, "semantic_rgb")    # train semantic
train_label_file  = os.path.join(root_dir, "train.csv") # train file
labels_dir        = os.path.join(root_dir, "labels") # train labels

Label = namedtuple('Label', [
				   'name', 
				   'id', 
				   'color'])

labels = [ # name                      id       color
	Label(  'unlabeled'              ,  0  , (  0,  0,  0)),
	Label(  'dynamic'                , 0   , (111, 74,  0)),
	Label(  'ground'                 , 0   , ( 81,  0, 81)),
	Label(  'road'                   ,  1  , (128,  64,128)),
	Label(  'sidewalk'               ,  2  , (244, 35,232)),
	Label(  'parking'                ,  1  , (250,170,160)),
	Label(  'rail track'             ,  1  , (230,150,140)),
	Label(  'buildings'              ,  3  , ( 70, 70, 70)),
	Label(  'wall'                   ,  3  , (102,102,156)),
	Label(  'fence'                  ,  3  , (190,153,153)),
	Label(  'guard rail'             ,  3  , (180,165,180)),
	Label(  'billboards'             ,  5  , (250,170, 30)),
	Label(  'pole'                   ,  6  , (153,153,153)),
	Label(  'traffic light'          ,  7  , (250,170, 30)),
	Label(  'vegetation'             ,  8  , (107,142, 35)),
	Label(  'terrain'                ,  8  , (152,251,152)),
	Label(  'sky'                    ,  9  , ( 70,130,180)),
	Label(  'person'                 ,  10 , (220, 20, 60)),
	Label(  'rider'                  ,  10 , (255,  0,  0)),
	Label(  'car'                    ,  11 , (  0,  0,142)),
	Label(  'truck'                  ,  11 , (  0,  0, 70)),
	Label(  'bus'                    ,  12 , (  0, 60,100)),
	Label(  'train'                  ,  12 , (  0, 80,100)),
	Label(  'motorcycle'             , 11  , (  0,  0,230)),
    Label(  'bicycle'                , 11  , (119, 11, 32)),
    Label(  'license plate'          , 11  , (  0,  0,142))
	]

color2index = {}
index2color = {}
colors = []
id_list = []
for obj in labels:
		idx   = obj.id
		label = obj.name
		color = obj.color
		color2index[color] = idx
		index2color[idx] = color
		id_list.append(idx)
colors = np.array(colors)

for dir in [semantic_dir, labels_dir, images_dir, images2_dir]:
	if not os.path.exists(dir):
		os.makedirs(dir)


def parse_label():
	t = open(train_label_file, "w")
	t.write("img,label\n")
	
	
	for idx, name in enumerate(os.listdir(semantic_dir)):
		
		filename_img = os.path.join(images_dir , name)
		filename_sem = os.path.join(semantic_dir, name)
        
		frame = Image.open(filename_img)
		filename_img = os.path.join(images2_dir, name)
		frame.save(filename_img, 'png')

		frame = np.asarray(Image.open(filename_sem).convert('RGB'))
		height, weight, _ = frame.shape
		idx_mat = np.zeros((height, weight))
		for h in range(height):
			for w in range(weight):
				color = tuple(frame[h, w])
				try:
					index = color2index[color]
					idx_mat[h, w] = index
				except:
					# no index, assign to void
					idx_mat[h, w] = 0
		idx_mat = idx_mat.astype(np.uint8)

		label_name = os.path.join(labels_dir, name)

		Image.fromarray(idx_mat, mode='L').save(label_name+ '.png')
		t.write("{},{}\n".format(filename_img, label_name + '.png'))
	t.close()


if __name__ == '__main__':
	parse_label()
