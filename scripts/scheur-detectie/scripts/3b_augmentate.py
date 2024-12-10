# -*- coding: utf-8 -*-
"""
Data augmentation is applied to the k-fold dataset.
Both geometric and color augmenation is applied (firt geometric then color)
specify n_transforms in config.yml. 
if n_transforms = 5, 5x5=25 augmented images are created

@authors: 
    Joost Driebergen (HKV lijn in water)
    Jeroen Baars     (HHNK)
"""

import yaml
from pathlib import Path
from augment_functions.Augmentation import Augmentation
from helper_functions.functions import kfold_folder_buildr

#%%
# Load config file
with open("../config.yml", "r") as ymlfile:
    cfg = yaml.load(ymlfile, yaml.SafeLoader)

# Load config parameters
k_fold      = cfg["cross_val"]["k_folds"]
N_transform_geo = cfg["cross_val"]["n_transforms_geo"]
N_transform_rgb = cfg["cross_val"]["n_transforms_rgb"]

# Load subimage size from config
target_width  = cfg["image"]["subimage_width"]
target_height = cfg["image"]["subimage_height"]

# Load paths from config 
data_kfolds_fil = Path(cfg["folder"]["kfolds_folder"])
data_kfolds_geo = Path(cfg["folder"]["kfolds_geo"])
data_kfolds_rgb = Path(cfg["folder"]["kfolds_rgb"])

sub_filter  = Path(cfg["folder"]["sub_filter"])
sub_geo     = Path(cfg["folder"]["sub_geo"])
sub_rgb     = Path(cfg["folder"]["sub_rgb"])

# Create augmentation class
Augment = Augmentation(target_width, target_height)

# Parameters
splits = ['train','test','val']
extentie = '*.png'

for fold in range(1, k_fold+1):
    print('Processing fold {}'.format(fold))
    #Apply augmentation on each fold
    for split in splits:
        print('Split {}'.format(split))

        data_folder_img, data_folder_mask = kfold_folder_buildr(data_kfolds_fil, \
            sub_filter, fold, split)
        out_geotf_img, out_geotf_mask = kfold_folder_buildr(data_kfolds_geo, \
            sub_geo, fold, split, make_dir=True)
        out_rgbtf_img, out_rgbtf_mask = kfold_folder_buildr(data_kfolds_rgb, \
            sub_rgb, fold, split, make_dir=True)        
        
        print('Start geo augmentation')
        Augment.apply_transform(Augment.geotransform, data_folder_img, \
            data_folder_mask, out_geotf_img, out_geotf_mask, N_transform_geo, extentie)
        
        print('Start color augmentation')
        Augment.apply_transform(Augment.colortransform, out_geotf_img, \
            out_geotf_mask, out_rgbtf_img, out_rgbtf_mask, N_transform_rgb, extentie)