"""
@author: Adugna Mullissa
@software: Spyder
@file: test.py
@time: 2020/02/01 16:54
@desc: @desc: This script run a semantic segmentation prediction task on images found in
 data folder. The script is written for multi-band tiff images already prepared
 as patches. 
 This script is modified from https://github.com/lsh1994/keras-segmentation.
"""
###############################################################################
import LoadBatches
#from keras.models import load_model
from Models import FCN
import glob
import cv2
import numpy as np
import random
import tifffile
import matplotlib.pyplot as plt
plt.style.use("ggplot")
###############################################################################
n_classes = 13
key = "fcn"
method = {
    "fcn": FCN.FCN_DK3}

images_path = "data/Test_image/"
segs_path = "data/Test_label/"

input_height = 125
input_width = 125

colors = [
    (random.randint(
        0, 255), random.randint(
            0, 255), random.randint(
                0, 255)) for _ in range(n_classes)]

##########################################################################


def label2color(colors, n_classes, seg):
    seg_color = np.zeros((seg.shape[0], seg.shape[1], 3))
    for c in range(n_classes):
        seg_color[:, :, 0] += ((seg == c) *
                               (colors[c][0])).astype('uint8')
        seg_color[:, :, 1] += ((seg == c) *
                               (colors[c][1])).astype('uint8')
        seg_color[:, :, 2] += ((seg == c) *
                               (colors[c][2])).astype('uint8')
    seg_color = seg_color.astype(np.uint8)
    return seg_color


def getcenteroffset(shape, input_height, input_width):
    short_edge = min(shape[:2])
    xx = int((shape[0] - short_edge) / 2)
    yy = int((shape[1] - short_edge) / 2)
    return xx, yy


images = sorted(
    glob.glob(
        images_path +
        "*.tif") +
    glob.glob(
        images_path +
        "*.png") +
    glob.glob(
        images_path +
        "*.jpeg"))
segmentations = sorted(glob.glob(segs_path + "*.tif") +
                       glob.glob(segs_path + "*.png") + glob.glob(segs_path + "*.jpeg"))


# m = load_model("output/%s_model.h5" % key)
m = method[key](13, 125, 125)  
m.load_weights("output/%s_model.h5" % key)

for i, (imgName, segName) in enumerate(zip(images, segmentations)):

    print("%d/%d %s" % (i + 1, len(images), imgName))

    #im = cv2.imread(imgName, 1) #Uncomment when using RGB images
    im = tifffile.imread(imgName) #comment out when using RGB imaes
    im = np.swapaxes(im,0,1) #comment out when using RGB imaes
    im = np.swapaxes(im,1,2) #comment out when using RGB imaes
    # im=cv2.resize(im,(input_height,input_width))
    xx, yy = getcenteroffset(im.shape, input_height, input_width)
    #im = im[xx:xx + input_height, yy:yy + input_width, :]

    seg = tifffile.imread(segName) #comment out when using RGB imaes
    #seg = cv2.imread(segName, 0) #Uncomment when using an RGB image
    # seg= cv2.resize(seg,interpolation=cv2.INTER_NEAREST)
    #seg = seg[xx:xx + input_height, yy:yy + input_width]
    
    pr = m.predict(np.expand_dims(LoadBatches.getImageArr(im), 0))[0]
    pr = pr.reshape((input_height, input_width, n_classes)).argmax(axis=2)

    #cv2.imshow("img", im)
    cv2.imshow("seg_predict_res", label2color(colors, n_classes, pr))
    cv2.imshow("seg", label2color(colors, n_classes, seg))
    cv2.waitKey()

