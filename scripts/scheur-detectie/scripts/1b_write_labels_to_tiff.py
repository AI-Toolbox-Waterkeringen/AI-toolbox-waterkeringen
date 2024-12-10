# -*- coding: utf-8 -*-
"""
From supervisely we export .png files with labels (mask_machine)
In this script, the mask png files are converted back to .tiff files based
on their reference rgb tiff-file.
Result is written to {tiffiles_mask}

@authors: 
    Joost Driebergen (HKV lijn in water)
    Jeroen Baars     (HHNK)
"""
from pathlib import Path
import rasterio as rio
import numpy as np
import yaml

# Load config file
with open("../config.yml", "r") as ymlfile:
    cfg = yaml.load(ymlfile, yaml.SafeLoader)

tiffiles      = Path(cfg['folder']['tiffiles'])
maskfiles     = Path(cfg['folder']['input_folder']) / cfg['folder']['input_masks']
mask_tiff_out = Path(cfg['folder']['tiffiles_mask'])


for maskfile in maskfiles.glob('*.png'):
    tiffile = tiffiles / (maskfile.stem + '.tif')

    img = rio.open(str(maskfile))
    img = img.read([1,2,3])
    img = img.astype('uint8')

    img_write = np.empty((4,img.shape[1],img.shape[2]))
    img_write[0:3,:,:] = img
    img_write[-1,img[0,:,:] == 1] = 1
    img_write = img_write * 255
    img_write = img_write.astype('uint8')

    outfile = mask_tiff_out / (tiffile.stem + '_mask.tif')
    
    print(outfile)

    with rio.open(tiffile) as naip:
    #open georeferenced.tif for writing
        with rio.open(
            outfile,
            'w',
            driver='GTiff',
            count=img_write.shape[0],
            height=img_write.shape[1],
            width=img_write.shape[2],
            dtype=img_write.dtype,
            crs=naip.crs,
            transform=naip.transform,
            compress='lzw'
            ) as dst:
                dst.write(img_write.astype(rio.uint8))
        
        # dat = rio.open(outfile)
        # profile = dat.profile.copy()
        # profile.update(compress='lzw')
        
        # #Compression
        # with rio.open(outfile, 'w', **profile) as dst:
        #     for ji, window in dat.block_windows(1):
        #         dst.write(dat.read(window=window), window=window)