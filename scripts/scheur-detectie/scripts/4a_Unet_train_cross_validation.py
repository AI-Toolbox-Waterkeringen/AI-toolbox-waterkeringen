# -*- coding: utf-8 -*-
"""
Script to train a K-fold cross-validated dataset where the following hyperparameters have to be optimized:
- EPOCHS
- batchsize
- Learning rate
The script exports a csv file with the results of the optimization (Optimization_results_Kfold={k_fold}.csv)
A plot of the training/validation loss is written to {out_folder} (specify in config.yml)

@authors: 
    Joost Driebergen (HKV lijn in water)
    Jeroen Baars     (HHNK)
"""

import yaml
import argparse
import pandas as pd
from pathlib import Path
from unet_functions.unet_cross_validation import unet_kfold_prediction

#Load argument parser arguments. Also dataset is parsed with help set to "dataset" so it is properly mounted.
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='dataset')
parser.add_argument('--learning_rate', type=float, nargs='+')
parser.add_argument('--batch_size', type=int, nargs='+')
parser.add_argument('--epochs', type=int, nargs='+')
args = parser.parse_args()

#Load arguments into variables
dataset         = Path(args.dataset)
epochs          = args.epochs
batches         = args.batch_size
learning_rate   = args.learning_rate

# Load config file
with open("config.yml", "r") as ymlfile:
    cfg = yaml.load(ymlfile, yaml.SafeLoader)

#Load relative paths from configuration file
k_fold     = cfg['cross_val']['k_folds']
weights    = cfg['unet']['petrained_weights']
output     = Path(cfg['folder']['output_folder'])

df_results = pd.DataFrame(columns = ['EPOCHS','batch','LR','fold','train_dice','val_dice','test_dice'])
results_location  = output / 'Optimization_results_Kfold={}.csv'.format(k_fold)
df_results.to_csv(results_location, index=False)

for epoch in epochs:
    for lr in learning_rate:
        for batch in batches:
            print(epoch,lr,batch)
            df_results = unet_kfold_prediction(k_fold,lr,epoch,batch,weights,df_results,output,dataset)
            df_results.to_csv(results_location, mode='a', index=False, header=False)