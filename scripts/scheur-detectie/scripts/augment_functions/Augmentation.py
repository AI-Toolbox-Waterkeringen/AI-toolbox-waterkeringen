# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 13:45:44 2021

@Organization: HHNK, HKV
@author: Joost Driebergen, Jeroen Baars

Functies om data augmentation toe te passen op images

"""
import cv2
import random
import numpy as np
import albumentations as A

random.seed(30)
np.random.seed(30)

class Augmentation:
    def __init__(self,img_width,img_height):
        self.width = img_width
        self.height = img_height
    
    def geotransform(self,image,mask):
        transform = A.Compose([A.augmentations.transforms.Flip(p=0.5),
                               A.augmentations.geometric.rotate.RandomRotate90(p=0.5),
                               A.augmentations.crops.transforms.RandomResizedCrop(self.height, self.width, scale=(0.08, 1.0), ratio=(0.75, 1.3), interpolation=1, always_apply=False, p=0.5)                               
                               ])
        transformed = transform(image=image,mask = mask)
        return transformed['image'],transformed['mask']
    
    def colortransform(self,image,mask):
        transform = A.Compose([
            # A.augmentations.transforms.Blur (blur_limit=7, always_apply=False, p=0.5),
            A.augmentations.transforms.ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.5),
            A.augmentations.transforms.GaussNoise (var_limit=(10.0, 50.0), mean=0, per_channel=True, always_apply=False, p=0.5),
            # A.augmentations.transforms.HueSaturationValue (hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=False, p=0.5),
            # A.augmentations.transforms.RandomBrightnessContrast (brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=0.5)
        ])
        transformed = transform(image=image, mask = mask)
        return transformed['image'],transformed['mask']
        
    def check_unique_augment(self,new_augment,history_augment):
        """
        Function to check if a generated augmentation is unique.
        """
        for n in range(len(history_augment)):
            if (new_augment == history_augment[n]).all():
                #print('Not unique')
                return False            
        return True

    def apply_transform(self,transform, image_folder_in, mask_folder_in, image_folder_out, mask_folder_out, N_transform,extentie):
        """
        Function to apply an augmentation transformation on a set of images (in image_folder)
        An ammount of augmentations is applied and the result is written to disk.
        """
        for img in sorted(image_folder_in.glob(extentie)):
            image = cv2.imread(str(img))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if (image.max() == image.mean()):
                continue
            mask = cv2.imread(str(mask_folder_in / img.name),flags=(cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH))
            list_hist = []
            list_hist.append(image)
            processed = 0
            while not(processed == N_transform):
                image_transform,mask_transform = transform(image=image,mask=mask)
                if self.check_unique_augment(image_transform,list_hist):            
                    list_hist.append(image_transform)
                    imname_augmented = image_folder_out / (img.stem + '_' + str(processed) + img.suffix)
                    imname_mask      = mask_folder_out / (img.stem + '_' + str(processed) + img.suffix)
                    cv2.imwrite(str(imname_augmented),image_transform)
                    cv2.imwrite(str(imname_mask),mask_transform)
                    processed +=1
        
    
