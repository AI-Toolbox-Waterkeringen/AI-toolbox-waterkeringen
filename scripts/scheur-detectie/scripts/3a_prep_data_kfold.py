# -*- coding: utf-8 -*-
"""
A k-fold dataset is created in this script.
Specify split in config.yml
i.e. with a split of 0.1 trainingssize is 0.8 and test/val both 0.1

@authors: 
    Joost Driebergen (HKV lijn in water)
    Jeroen Baars     (HHNK)
"""

import yaml
from pathlib import Path
from helper_functions.functions import load_data_cross, copy_images, kfold_folder_buildr

# Load config file
with open("../config.yml", "r") as ymlfile:
    cfg = yaml.load(ymlfile, yaml.SafeLoader)

# Load config parameters
k_fold      = cfg["cross_val"]["k_folds"]
split       = cfg["cross_val"]["split"]

# Load paths from config 
data_filter = Path(cfg["folder"]["filter_folder"])
data_kfolds = Path(cfg["folder"]["kfolds_folder"])
sub_filter  = Path(cfg["folder"]["sub_filter"])

# Create K-folds split for the data
(train_x_kfold, train_y_kfold), (valid_x_kfold, valid_y_kfold), (test_x_kfold, test_y_kfold) \
    = load_data_cross(data_filter, k_fold, split)

for fold in range(1, k_fold+1):
    print('Processing fold {}'.format(fold))
    
    img_train, mask_train = kfold_folder_buildr(data_kfolds, sub_filter, \
        fold, "train", make_dir=True)
    img_test , mask_test  = kfold_folder_buildr(data_kfolds, sub_filter, \
        fold, "test" , make_dir=True)
    img_val  , mask_val   = kfold_folder_buildr(data_kfolds, sub_filter, \
        fold, "val"  , make_dir=True)
    
    copy_images(train_x_kfold[fold-1], train_y_kfold[fold-1], img_train, mask_train)
    copy_images(valid_x_kfold[fold-1], valid_y_kfold[fold-1], img_val,   mask_val)
    copy_images(test_x_kfold[fold-1],  test_y_kfold[fold-1],  img_test,  mask_test)