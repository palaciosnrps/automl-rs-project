import os
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from skimage.transform import resize
from PIL import Image
import tifffile as tiff
import tensorflow as tf
import tensorflow_datasets as tfds
import sklearn.model_selection
from tensorflow.keras.utils import to_categorical

DATA_DIR='/scratch/palacios/data'

def create_dataset(NAME='eurosat_rgb',SPLIT=0.2):
    """This function loads and transform the datasets to create
    training and test data
    #Arguments:
        NAME (String): dataset to load
        SPLIT (Float): percentage of the data that will be used to create thetest dataset
        INPUT_SIZE (Tuple): image shape (height, width, channels) 
    TO DO: CHANGE STATIC DATA LOCATIONS
    """
    DATA_NAMES={
        'eurosat_rgb':'eurosat/rgb',
        'eurosat_all':'eurosat/all',
        'bigearthnet_rgb':'bigearthnet/rgb',
        'bigearthnet_all':'bigearthnet/all',
        'brazildam':'brazildam_sentinel',
        'coffee_scenes': 'coffee_scenes',
        'savana':'savana_scenes',
        'croptype': 'crop_type'
    }
    data_tfds=tfds.as_numpy(tfds.load(DATA_NAMES[NAME], data_dir=DATA_DIR,batch_size=-1, as_supervised=True,shuffle_files=True))
    if NAME in ['eurosat_rgb','eurosat_all','bigearthnet_rgb','bigearthnet_all']:    
        X_train,  X_test, y_train, y_test = sklearn.model_selection.train_test_split(data_tfds['train'][0],data_tfds['train'][1], test_size=SPLIT)
    else:
        def normalize(imgi):
            img=resize(imgi,INPUT_SIZE)
            min = img.min()
            max = img.max()
            x = 2.0 * (img - min) / (max - min) - 1.0
            return x
        if NAME=='brazildam':
            X_train,  X_test, y_train, y_test = sklearn.model_selection.train_test_split(data_tfds['train_2019'][0],data_tfds['train_2019'][1], test_size=SPLIT)
        if NAME in ['coffee_scenes','savana']:
            X_train=[]
            y_train=[]
            X_test=data_tfds['fold5'][0]
            y_test=data_tfds['fold5'][1]
            for i in range(1,5):
                X_train.extend(data_tfds['fold'+str(i)][0])
                y_train.extend(data_tfds['fold'+str(i)][1])
    return X_train, X_test, y_train, y_test

x1,x2,y1,y2=create_dataset('coffee_scenes')
print(len(x1))
print(len(x2))
print(len(y1))
print(len(y2))
