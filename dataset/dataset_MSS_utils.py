from __future__ import print_function


from matplotlib import pyplot as plt
import numpy as np
import random    
import cv2
import scipy.misc
from PIL import Image
import os
from collections import namedtuple
from matplotlib import pyplot as plt
import re
from scipy.ndimage import generic_filter
from scipy import stats


def modal(P):
    """We receive P[0]..P[8] with the pixels in the 3x3 surrounding window"""
    mode = stats.mode(P)
    return mode.mode[0]



#############################
    # global variables #
#############################
root_dir          = "./../MSS/"
video_dir         = os.path.join(root_dir, "MSSdataset/")    # videos
data_dir          = os.path.join(root_dir, "data/")      # Data for train and test
images_dir        = os.path.join(data_dir, "images/")    # train label
label_dir         = os.path.join(data_dir, "labels/")    # train label
train_label_file  = os.path.join(data_dir, "trainCS.csv") # train file


Label = namedtuple('Label', [
                   'name', 
                   'id', 
                   'color'])

labels = [ # name                      id       color
    Label(  'unlabeled'             ,  0  , (  0,  0,   0)  ),
    Label(  'road'                  ,  1  , (  51,  22,   78)  ),
    Label(  'sidewalk2'              ,  2  , (  140,  3,  71)  ),
    Label(  'sidewalk'               ,  2  , (  255,  20,  147)  ),
    Label(  'buildings_2'            ,  3  , (  61,    70,  77)  ),
    Label(  'buildings (complements)',  3  , (  112,  128,  144) ),
    Label(  'billboards'             ,  8  , (  188,  143,  143) ),
    Label(  'pole'                   ,  6  , (  179,  169,  176) ),
    Label(  'traffic light'          ,  7  , (  255,  255,  0)   ),
    Label(  'vegetation_2'           ,  9  , (  13,  75,    8)   ),
    Label(  'vegetation_2'           ,  9  , (  45,  75,    45)   ),
    Label(  'sky'                    ,  11  , (  0,  191,  255)   ),
    Label(  'sky'                    ,  11  , (  96,  135,  142)   ),
    Label(  'person'                 ,  12 , (  133,  0,  0)     ),
    Label(  'person'                 ,  12 , (  162,  20,  36)     ),
    Label(  'person'                 ,  12 , (  255,  0,  0)     ),
    Label(  'car2'                   ,  14 , (  4,  20,  82)     ),
    Label(  'car2'                   ,  14 , (  0,  1,  65)     ),
    Label(  'car'                    ,  14 , (  0,  0,  128)     ),
    Label(  'bus_2'                  ,  16 , (  0,    68,  63)   )
    ]

color2index = {}
index2color = {}
colors = []
id_list = {}

for i, obj in enumerate(labels):
        idx   = obj.id
        label = obj.name
        color = obj.color
        colors.append(color)
        color2index[color] = idx
        index2color[idx] = color
        id_list[i] = idx
colors = np.array(colors)
for dir in [video_dir, data_dir, label_dir, images_dir]:
    if not os.path.exists(dir):
        os.makedirs(dir)
print(id_list)
def label_to_RGB(image):
    height, weight = image.shape

    rgb = np.zeros((height, weight, 3), dtype=np.uint8)
    for h in range(height):
        for w in range(weight):
            rgb[h,w,:] = index2color[image[h,w]]
    return rgb

def color_dist(color, colors):
    #print(color, colors)
    """rmean = (color[0]-colors[:,0])/2
    r = color[0]-colors[:,0]
    g = color[1]-colors[:,1]
    b = color[2]-colors[:,2]
    results = np.sqrt((((512+rmean)*r*r) //18) + 4*g*g + (((767-rmean)*b*b)//16))"""
    results = np.sqrt((color[0]-colors[:,0])**2+(color[2]-colors[:,2])**2+(color[1]-colors[:,1])**2)
    #print(results)
    argmin = np.argmin(results)
    if results[argmin] > 10:
        return 0
    return id_list[argmin]

def parse_label(frame):
    height, width, _ = frame.shape
    #print(frame.shape)
    matrix = np.swapaxes(np.swapaxes(frame, 0,2), 1,2)
    #print(frame.shape)
    size = height*width
    distance = []
    for color in colors:
        color = np.repeat(color, size).reshape((3, height, width))
        #print(color)
        #break
        dist = np.sum((color - matrix)**2,0)
        distance.append(dist)
    distance = np.array(distance)
    #print(distance.shape)
    idx_mat = np.argmin(distance, 0)
    
    idx_mat = np.vectorize(id_list.get)(idx_mat)
    #print(idx_mat)
    #idx_mat = generic_filter(idx_mat, modal, (3, 3))
    
    #plt.subplot(1,2,1)
    #plt.imshow(idx_mat)
    #plt.subplot(1,2,2)
    #plt.imshow(frame)
    #plt.show()
    #cv2.cvtColor(np.float32()/255, cv2.COLOR_RGB2HSV)
    return idx_mat


def trim_video():
    
    
    for category in os.listdir(video_dir):
        
        category_dir = os.path.join(video_dir, category)
        img_category = os.path.join(images_dir, category)
        sem_category = os.path.join(label_dir, category)

        if not os.path.exists(img_category):
            os.makedirs(img_category)
            os.makedirs(sem_category)

        for amount_cars in os.listdir(category_dir):
            curriculum_dir = os.path.join(category_dir, amount_cars)
            img_category_cv = os.path.join(img_category, amount_cars)
            sem_category_cv = os.path.join(sem_category, amount_cars)

            if not os.path.exists(img_category_cv):
                os.makedirs(img_category_cv)
                os.makedirs(sem_category_cv)
                os.makedirs(sem_category_cv+"/unprocessed/")

            for video in os.listdir(curriculum_dir):
                print(video)
                if "RGB" not in video or os.path.exists(os.path.join(sem_category_cv, video[:-8] + "_0.png")):
            
                    continue
                #print(os.path.join(sem_category_cv, video[:-8] + "_0.jpg"))
                filename_dat = os.path.join(curriculum_dir, video)
                filename_sem = os.path.join(curriculum_dir, video[:-7]+ "Semantica.avi")
                
                i = 0
                cap= cv2.VideoCapture(filename_sem)

                while(cap.isOpened()):
                    
                    ret, frame = cap.read()
                    if ret == False:
                        break
                    sem_name = os.path.join(sem_category_cv, video[:-8] + "_"+ str(i) +".png")
                    if os.path.exists(sem_name):# or i%2!=0: #i%11 != 0: or
                        i += 1
                        continue
                    cv2.imwrite(sem_name, frame)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    #labels = parse_label(frame)
                    #image_name = os.path.join(sem_category_cv, video[:-8] + "_" + str(i))
                    #img = Image.fromarray(label_to_RGB(labels), 'RGB')
                   
                    #real_name = os.path.join(img_category_cv, video[:-8] + "_" +str(i))
                    #np.save(image_name, labels)
                    #img.save(image_name + '.jpg')
                    #cv2.imwrite(os.path.join(sem_category_cv,"unprocessed/" + video[:-8] + "_" + str(i)) + '.jpg',cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    #total.write("{},{}\n".format(real_name+ '.jpg', image_name + '.npy'))
                    i+=1
                    
                        
                cap.release()
                cv2.destroyAllWindows()
                
                i = 0
                cap= cv2.VideoCapture(filename_dat)
                while(cap.isOpened()):
                    ret, frame = cap.read()
                    if ret == False:
                        break
                    if os.path.exists(os.path.join(sem_category_cv, video[:-8] + "_"+ str(i) +".png")):
                        image_name = os.path.join(img_category_cv, video[:-8] + "_" +str(i))
                        cv2.imwrite(image_name + '.jpg',frame)
                    i+=1
         
                cap.release()
                cv2.destroyAllWindows()
                

def create_csv():
    total = open(train_label_file, "w")
    total.write("img,label\n")
    for category in os.listdir(images_dir):
        
        img_category = os.path.join(images_dir, category)
        sem_category = os.path.join(label_dir, category)

        for amount_cars in os.listdir(img_category):
            if not os.path.isfile(os.path.join(data_dir, str(amount_cars)+"CS.csv")):
                cv = open(os.path.join(data_dir, str(amount_cars) + "CS.csv"), "w")
                cv.write("img, label\n")
            else:
                cv = open(os.path.join(data_dir, str(amount_cars) + "CS.csv"), "+w")
            img_category_cv = os.path.join(img_category, amount_cars)
            sem_category_cv = os.path.join(sem_category, amount_cars)
            for img in os.listdir(img_category_cv):
                real_name = os.path.join(img_category_cv, img)
                label_name = os.path.join(sem_category_cv, img)
                if not os.path.isfile(label_name[:-4]+"CS.png"):
                    img = np.array(Image.open(label_name).convert("RGB"))
                    labels = parse_label(img)
                    np.savez_compressed(label_name[:-4]+"CS", labels)
                if os.path.isfile(real_name) and os.path.isfile(label_name[:-4]+"CS.npz"):
                    total.write("{},{}\n".format(real_name, label_name[:-4]+"CS.npz"))
                    cv.write("{},{}\n".format(real_name, label_name[:-4] + "CS.npz"))
            cv.close()
    total.close()
                
if __name__ == '__main__':
    trim_video()
    create_csv()
    
    
