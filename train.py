"""
@author: Adugna Mullissa
@software: Spyder
@file: train.py
@time: 2020/02/01 16:54
@desc: This script run a semantic segmentation training task on images found in
 data folder. The script is written for multi-band tiff images already prepared
 as patches. 
 This script is modified from https://github.com/lsh1994/keras-segmentation.
"""

##############################################################################
from keras.callbacks import ModelCheckpoint, TensorBoard
import LoadBatches
from Models import FCN
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import math
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import numpy as np

#############################################################################
train_images_path = "data/images_train/"
train_segs_path = "data/label_train/"
train_batch_size = 32
n_classes = 13

epochs = 150

input_height = 125
input_width = 125


val_images_path = "data/images_test/"
val_segs_path = "data/label_test/"
val_batch_size = 32

key = "fcn"


method = {
    "fcn": FCN.FCN_DK3}

m = method[key](n_classes, input_height=input_height, input_width=input_width)

m.compile(
    loss='categorical_crossentropy',
    optimizer='adadelta',
    metrics=['acc'])
m.summary()

G = LoadBatches.imageSegmentationGenerator(train_images_path,
                                           train_segs_path, train_batch_size, n_classes=n_classes, input_height=input_height, input_width=input_width)

G_test = LoadBatches.imageSegmentationGenerator(val_images_path,
                                                val_segs_path, val_batch_size, n_classes=n_classes, input_height=input_height, input_width=input_width)

checkpoint = ModelCheckpoint(
    filepath="output/%s_model.h5" %
    key,
    monitor='acc',
    mode='auto',
    save_best_only='True')
tensorboard = TensorBoard(log_dir='output/log_%s_model' % key)

result = m.fit_generator(generator=G,
                steps_per_epoch=math.ceil(367. / train_batch_size),
                epochs=epochs, callbacks=[checkpoint, tensorboard],
                verbose=2,
                validation_data=G_test,
                validation_steps=8,
                shuffle=True)

plt.figure(figsize=(8, 8))
plt.title("Learning curve")
plt.plot(result.history["acc"], label="acc")
plt.plot(result.history["val_acc"], label="val_acc")
plt.plot( np.argmax(result.history["val_acc"]), np.max(result.history["val_acc"]), marker="x", color="r", label="best model")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend();

plt.figure(figsize=(8, 8))
plt.title("Learning curve")
plt.plot(result.history["loss"], label="loss")
plt.plot(result.history["val_loss"], label="val_loss")
plt.plot( np.argmin(result.history["val_loss"]), np.min(result.history["val_loss"]), marker="x", color="r", label="best model")
plt.xlabel("Epochs")
plt.ylabel("log_loss")
plt.legend();
