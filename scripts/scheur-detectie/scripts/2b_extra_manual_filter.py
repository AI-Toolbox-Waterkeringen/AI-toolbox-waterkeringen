"""
Optional script
Images specified are removed from the dataset  
Specify images_manual_select.txt in config.yml

@authors: 
    Joost Driebergen (HKV lijn in water)
    Jeroen Baars     (HHNK)
"""

import os, pickle, yaml
from glob import glob
from pathlib import Path
from helper_functions.functions import folder_buildr

# Copied folder and deleted all images with black or unexpected visuals
# Load image names of folder and save to create file that can be used by others
# images  = sorted(glob(os.path.join("../../dronebeelden_hhnk/subimages_256_filtered/img_manual/*.png")))
# images = [img.split("\\")[-1] for img in images]
# with open("images_manual_select.txt", "wb") as fp:
#    pickle.dump(images, fp)

# To reproduce manual removal of images on other machines
with open("images_manual_select.txt", "rb") as fp:
   images_manual = pickle.load(fp)

# Load config file
with open("../config.yml", "r") as ymlfile:
    cfg = yaml.load(ymlfile, yaml.SafeLoader)

# Load paths from config
data_filter = Path(cfg["folder"]["filter_folder"])
filter_img, filter_masks = folder_buildr(data_filter)

images  = [os.path.basename(x) for x in sorted(glob(os.path.join(filter_img, "*.png")))]
diff = list(set(images) - set(images_manual))
print("Removed {} of the {} images because of manual filtering".format(len(diff), len(images)))

for file in diff:
   os.remove(filter_img / file)
   os.remove(filter_masks / file)