import numpy as np
from dataset.Cityscapes_utils import idx2color as index2color
# borrow functions and modify it from https://github.com/Kaixhin/FCN-semantic-segmentation/blob/master/main.py
# Calculates class intersections over unions
def iou(pred, target):
    ious = []
    known = (target != 0)
    for i in range(1, 20):
        pred_inds = (pred == i)
        pred_inds = pred_inds*known
        target_inds = (target == i)
        intersection = (pred_inds*target_inds).sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / max(union, 1))
        # print("cls", cls, pred_inds.sum(), target_inds.sum(), intersection, float(intersection) / max(union, 1))
    return ious


def pixel_acc(pred, target):
    correct = (pred == target).sum()
    total   = (target == target).sum() - (target == 0).sum()
    return correct / total

def label_to_RGB(image):

    image = image.squeeze()
    height, weight = image.shape

    rgb = np.zeros((height, weight, 3))
    for h in range(height):
        for w in range(weight):
            try:
                rgb[h,w,:] = index2color[image[h,w]]
            except:
                rgb[h,w,:] = (0, 0, 0)
    return rgb.astype(np.uint8)
