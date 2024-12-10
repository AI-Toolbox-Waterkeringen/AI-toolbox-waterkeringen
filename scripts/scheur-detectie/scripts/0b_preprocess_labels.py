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
import numpy as np
from pathlib import Path
from helper_functions.functions import folder_buildr

# Load config file
with open("config.yml", "r") as ymlfile:
    cfg = yaml.load(ymlfile, yaml.SafeLoader)

# Load paths from config
input_masks = cfg["folder"]["input_masks"]
data_input  = Path(cfg["folder"]["input_folder"])

# Setup folders and make the directories
img, masks = folder_buildr(data_input, masks_dir=input_masks)

print(img, masks)

def read_image(path, IMAGE_SIZE):
    if not (type(path) == str):
        path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (IMAGE_SIZE, IMAGE_SIZE))
    x = x / 255.0
    x = x.astype(np.float32)
    return x

for mask_filter in glob.glob(str(masks/'*.png')):

    if "ilpendam" in str(mask_filter) or "jisp" in str(mask_filter):
        x = read_image(mask_filter, 4000)
        x = x*255
        x_temp = np.where(x==1,0,x)
        x_temp = np.where(x==2,1,x_temp)    
    else:
        x = read_image(mask_filter, 4096)
        x = x*255
        x_temp = np.where(x==2,0,x)

    print(np.unique(x_temp, return_counts=True))
    cv2.imwrite(mask_filter,x_temp)    