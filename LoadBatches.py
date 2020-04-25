import numpy as np
#import cv2
import glob
import itertools
import random
import tifffile


def getImageArr(im):
    img = im.astype(np.float32)
    return img


def getSegmentationArr(seg, nClasses, input_height, input_width):

    seg_labels = np.zeros((input_height, input_width, nClasses))

    for c in range(nClasses):
        seg_labels[:, :, c] = (seg == c).astype(int)

    seg_labels = np.reshape(seg_labels, (-1, nClasses))
    return seg_labels


def imageSegmentationGenerator(images_path, segs_path, batch_size,
                               n_classes, input_height, input_width):

    assert images_path[-1] == '/'
    assert segs_path[-1] == '/'

    images = sorted(glob.glob(images_path + "*.tif") +
                    glob.glob(images_path + "*.png") + glob.glob(images_path + "*.jpeg"))

    segmentations = sorted(glob.glob(segs_path + "*.tif") +
                           glob.glob(segs_path + "*.png") + glob.glob(segs_path + "*.jpeg"))

    zipped = itertools.cycle(zip(images, segmentations))

    while True:
        X = []
        Y = []
        for _ in range(batch_size):
            im, seg = zipped.__next__()
            #im = cv2.imread(im, -1) #uncomment when using RGB images
            im = tifffile.imread(im) #comment out when using RGB imaes
            im = np.swapaxes(im,0,1) #comment out when using RGB imaes
            im = np.asarray(im)
            #seg = cv2.imread(seg, 0) #uncomment when using RGB images
            seg = tifffile.imread(seg) #comment out when using RGB imaes

            assert im.shape[:2] == seg.shape[:2]

            assert im.shape[0] >= input_height and im.shape[1] >= input_width

            xx = random.randint(0, im.shape[0] - input_height)
            yy = random.randint(0, im.shape[1] - input_width)

            im = im[xx:xx + input_height, yy:yy + input_width]
            seg = seg[xx:xx + input_height, yy:yy + input_width]

            X.append(getImageArr(im))
            Y.append(
                getSegmentationArr(
                    seg,
                    n_classes,
                    input_height,
                    input_width))

        yield np.array(X), np.array(Y)


if __name__ == '__main__':
    G = imageSegmentationGenerator("data/images_train/",
                                   "data/label_train/", batch_size=64, n_classes=13, input_height=125, input_width=125)
    x, y = G.__next__()
    print(x.shape, y.shape)
