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
from pathlib import Path
from helper_functions.functions import folder_buildr, write_image_to_subimages

# Load config file
with open("../config.yml", "r") as ymlfile:
    cfg = yaml.load(ymlfile, yaml.SafeLoader)

# Load subimage size from config
target_width = cfg["image"]["subimage_width"]
target_height = cfg["image"]["subimage_height"]

# Load paths from config
input_masks = cfg["folder"]["input_masks"]
data_input = Path(cfg["folder"]["input_folder"])
data_subimg = Path(cfg["folder"]["subimg_folder"])

# Setup folders and make the directories
img, masks = folder_buildr(data_input, masks_dir=input_masks)
sub_img, sub_masks = folder_buildr(data_subimg, make_dir=True)

print("Folders: \n", img, "\n", masks, "\n")

for img_file, mask_file in zip(sorted(img.glob("*.png")), sorted(masks.glob("*.png"))):
    print(img_file, "\n", mask_file)
    img = cv2.imread(str(img_file))
    mask = cv2.imread(str(mask_file))
    write_image_to_subimages(
        img, img_file.stem, sub_img, target_width, target_height, multiply=1, dpi=None
    )
    write_image_to_subimages(
        mask,
        mask_file.stem,
        sub_masks,
        target_width,
        target_height,
        multiply=255,
        dpi=None,
    )
