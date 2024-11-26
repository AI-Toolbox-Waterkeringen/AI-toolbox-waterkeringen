# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 10:42:50 2021

@author: driebergen
"""


import os
import yaml
import pandas as pd
import tensorflow as tf
from glob import glob
from pathlib import Path
from tensorflow.keras.metrics import Recall, Precision
from unet_functions.unet_utils import unet_model,tf_dataset,plot_loss
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Load config file
with open("config.yml", "r") as ymlfile:
    cfg = yaml.load(ymlfile, yaml.SafeLoader)

SUBSET = cfg["folder"]["sub_rgb"]
IMAGE_SIZE = cfg['image']['subimage_height']

def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    smooth = 1e-15
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def unet_kfold_prediction(k_fold,LR,EPOCH,batch,weights,df_results,out_folder,input_dataset):
    print('Applying {}-fold crossvalidation'.format(k_fold))
    print('Using LR = {}, epoch = {} and batch = {}'.format(LR,EPOCH,batch))

    for fold in range(1,k_fold + 1):
        #DS_WV_SegmentateScheurdetectie/05_K_fold_cv_rgb/fold_1/subimages_augmentated_rgb/train/
        folder_fold = input_dataset / f"05_K_fold_cv_rgb" / f"fold_{fold}" / SUBSET
        print('Fold {}, using folder {}'.format(fold,folder_fold))

        train_x = sorted(glob(os.path.join(folder_fold, "train/img/*.png")))
        train_y = sorted(glob(os.path.join(folder_fold, "train/masks/*.png")))
        valid_x = sorted(glob(os.path.join(folder_fold, "val/img/*.png")))
        valid_y = sorted(glob(os.path.join(folder_fold, "val/masks/*.png")))
        test_x  = sorted(glob(os.path.join(folder_fold, "test/img/*.png")))
        test_y  = sorted(glob(os.path.join(folder_fold, "test/masks/*.png"))) 

        train_dataset = tf_dataset(train_x, train_y, batch=batch)
        valid_dataset = tf_dataset(valid_x, valid_y, batch=batch)
        test_dataset = tf_dataset(test_x, test_y, batch=batch)
        
        print("** Begin of data load checks **")
        print(input_dataset)
        print(os.listdir(input_dataset))
        print(os.path.join(folder_fold, "train/img/*.png"))
        print(len(train_x))
        print(len(train_y))
        print("** End of data load checks **")

        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = unet_model(IMAGE_SIZE,weights) 

            opt = tf.keras.optimizers.Nadam(LR)
            metrics = [dice_coef, Recall(), Precision()]
            model.compile(loss=dice_loss, optimizer=opt, metrics=metrics, run_eagerly=True)
        
        # Train the model 
        callbacks = [
            # ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=0.000005, verbose=1),
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False, verbose=1, start_from_epoch=40)
        ]
        
        train_steps = len(train_x)//batch
        valid_steps = len(valid_x)//batch
        test_steps = len(test_x)//batch

        if len(train_x) % batch != 0:
            train_steps += 1
        if len(valid_x) % batch != 0:
            valid_steps += 1
        if len(test_x) % batch != 0:
            test_steps += 1
        


        history = model.fit(
            train_dataset,
            validation_data=valid_dataset,
            epochs=EPOCH,
            steps_per_epoch=train_steps,
            validation_steps=valid_steps,
            callbacks=callbacks
        )
        
        outfile = out_folder / Path('LR_{}_EPOCHS_{}_batch_{}_fold_{}.png'.format(LR,EPOCH,batch,fold))
        
        train_loss,val_loss = plot_loss(history,outfile)
        
        dice_train = 1-train_loss
        # dice_val = 1-val_loss
        
        results_valid = model.evaluate(valid_dataset, steps=valid_steps)
        results_test  = model.evaluate(test_dataset, steps=test_steps)
        
        dice_val = results_valid[1]
        dice_test = results_test[1]
        
        name_weights = f"BATCHSIZE_{batch}_LR_{LR}_EPOCH_{EPOCH}_Fold_{fold}"
        print('Found new best fit for fold {}:{}'.format(fold,name_weights))
        model.save(str(out_folder / name_weights))
        
        df_app = pd.DataFrame(
            {
                "EPOCHS": [EPOCH],
                "batch": [batch],
                "LR": [LR],
                "fold": [fold],
                "train_dice": [dice_train],
                "val_dice": [dice_val],
                "test_dice": [dice_test],
            }
        )
        df_results = pd.concat([df_results, df_app], ignore_index=True)
    
    return df_results






