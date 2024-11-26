# -*- coding: utf-8 -*-
"""
Images without label are filtered from the dataset.
A fraction of images without label is added to the dataset
Specify fraction_no_label in config.yml

@authors: 
    Joost Driebergen (HKV lijn in water)
    Jeroen Baars     (HHNK)
"""
import random, yaml
import cv2
from pathlib import Path
from helper_functions.functions import folder_buildr, copy_file

#List to be filled
masks_with_label, masks_without_label = [], []

# Load config file
with open("../config.yml", "r") as ymlfile:
    cfg = yaml.load(ymlfile, yaml.SafeLoader)

# Fraction of data that is added without labels in the masks
random_seed = cfg["seed"]
fraction_no_label = cfg["image"]["fraction_no_lbl"]

# Load paths from config
data_subimg = Path(cfg["folder"]["subimg_folder"])
data_filter = Path(cfg["folder"]["filter_folder"])

# Setup folders and make the directories
subimg_img, subimg_masks = folder_buildr(data_subimg)
filter_img, filter_masks = folder_buildr(data_filter, make_dir=True)

# Filter out image with empty masks
for source_mask in sorted(subimg_masks.glob('*.png')):
    print(str(source_mask).split("masks")[1].ljust(60), end='\r')
    im = cv2.imread(str(source_mask))
    if im.sum() > 0:
        masks_with_label.append(source_mask)
    else:
        masks_without_label.append(source_mask)

#Print output of filter
print('Total images with label: {} vs without label: {}'\
    .format(len(masks_with_label), len(masks_without_label)))

# Random seed needed before every use of random
# select_val = int(len(masks_with_label)*fraction_no_label)
# print('Adding {} images without label'.format(select_val))
# random.seed(random_seed)
masks_2_copy = masks_with_label
#  + random.sample(masks_without_label, select_val)

#Copy filtered selection to folder
for source_mask in masks_2_copy:
    print(str(source_mask).split("masks")[1].ljust(60), end='\r')
    copy_file(source_mask, subimg_img, filter_img)
    copy_file(source_mask, subimg_masks, filter_masks)