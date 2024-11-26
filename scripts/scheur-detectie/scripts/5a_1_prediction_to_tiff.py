# -*- coding: utf-8 -*-
"""
Preprocesses the export from supervisely (images and masks) to subimages
Specify subimage_height and subimage_width in config.yml

@authors: 
    Joost Driebergen (HKV lijn in water)
    Jeroen Baars     (HHNK)
"""
import cv2
import yaml
import glob
import argparse
import numpy as np
import rasterio as rio
import tensorflow as tf
from pathlib import Path
from rasterio.windows import Window
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from helper_functions.functions import folder_buildr

def read_image(path, IMAGE_SIZE):
    if not (type(path) == str):
        path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (IMAGE_SIZE, IMAGE_SIZE))
    x = x / 255.0
    # x = x.astype(np.float32)
    return x

def dice_np(im1, im2):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    if (im1.sum() + im2.sum()) == 0:
        return 1
    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2.0 * intersection.sum() / (im1.sum() + im2.sum())

#Load argument parser arguments. Also dataset is parsed with help set to "dataset" so it is properly mounted.
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='dataset')
parser.add_argument('--learning_rate', type=float, nargs='+')
parser.add_argument('--batch_size', type=int, nargs='+')
parser.add_argument('--epochs', type=int, nargs='+')
parser.add_argument('--inputtiff', type=str, nargs='+')
args = parser.parse_args()

#Load arguments into variables
dataset         = Path(args.dataset)
epochs          = args.epochs
batches         = args.batch_size
learning_rate   = args.learning_rate
inputtiff         = args.inputtiff[0]

# Model V2 - 2023 with Wetterskip
folds = {4:{'EPOCHS':150,'batch':80,'LR':0.0001}}
fold  =  4    
weights_folder = "saved_weights/model_v2_plateau/" + 'BATCHSIZE_{}_LR_{}_EPOCH_{}_Fold_{}'.\
                    format(folds[fold]['batch'],folds[fold]['LR'],folds[fold]['EPOCHS'],fold)

def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    smooth = 1e-15
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

model = load_model(weights_folder,custom_objects={"dice_loss": dice_loss},compile=False)

# Load config file
with open("config.yml", "r") as ymlfile:
    cfg = yaml.load(ymlfile, yaml.SafeLoader)

print(inputtiff)

file_location = dataset / (inputtiff+".tif")
label_location = Path("outputs/" + inputtiff + "_label.tif")

# # Count=1 is for one band in mask
raster_file = rio.open(file_location)
profile = raster_file.profile.copy()
profile.update(count=1)

# profile.update(count=1)
with rio.open(str(label_location), 'w', **profile) as out_file:
    out_file.write(np.zeros([1,1,1]),window=Window.from_slices((0, 1), (0, 1)))

total = ((profile['height']-256)/256) * ((profile['width']-256)/256)

for idx in range(0,profile['height']-256,256):
    for idy in range(0,profile['width']-256,256):
        print(idx, " of ", total, end='\r')
        wdw = Window.from_slices((idx, idx+256), (idy, idy+256))

        with rio.open(str(file_location), 'r') as out_file:
            x = out_file.read(window=wdw)

        x = x[:3,:,:]
        x = np.rollaxis(x,0,3)
        
        threshold = 0.8
        
        y_pred = model.predict(np.expand_dims(x/255, axis=0))[0] 
        y_pred_threshold = np.where((y_pred*1000)>(threshold), 1, 0)
        
        with rio.open(str(label_location), 'r+', **profile) as out_file:
            out_file.write(y_pred_threshold.reshape(1,256,256), window=wdw)