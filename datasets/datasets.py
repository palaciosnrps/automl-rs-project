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

DATA_DIR='/data/s2105713/data'

def create_dataset(NAME='eurosat_rgb',SPLIT=0.2,INPUT_SIZE=(64,64,3)):
    """This function loads and transform the datasets to create
    training and test data
    #Arguments:
        NAME (String): dataset to load
        SPLIT (Float): percentage of the data that will be used to create thetest dataset
        INPUT_SIZE (Tuple): image shape (height, width, channels) 
    TO DO: CHANGE STATIC DATA LOCATIONS
    """
    DATA_NAMES={
        "eurosat_rgb":"eurosat/rgb",
        "eurosat_all":"eurosat/all",
        "bigearthnet_rgb":"bigearthnet/rgb",
        "bigearthnet_all":"bigearthnet/all",
        "brazildam":"/data/s2105713/data/BrazilDam/2019",
        "coffee_scenes": "/data/s2105713/data/brazilian_coffee_scenes/",
        "savana":"/data/s2105713/data/Brazilian_Cerrado_Savana_Scenes_Dataset/"
    }
    if NAME in ['eurosat_rgb','eurosat_all','bigearthnet_rgb']:
        data_tfds=tfds.load(DATA_NAMES[NAME], data_dir=DATA_DIR,split='train', as_supervised=True,shuffle_files=True)
        all_images=[]
        all_labels=[]
        def createdt(i,l):
            all_images.append(i)
            all_labels.append(l)
            #all_images=np.vstack([all_images,i]) if all_images.size else i
            #all_labels=np.vstack([all_labels,l]) if all_labels.size else l
        Parallel(n_jobs=-1, verbose=1, prefer="threads")(delayed(createdt)(i,l) for i,l in tfds.as_numpy(data_tfds))
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(all_images, all_labels,test_size=SPLIT)
        X_train=np.array(X_train)
        y_train=np.array(y_train)
        X_test=np.array(X_test)
        y_test=np.array(y_test)
        return X_train, X_test, y_train, y_test
    else:
        def normalize(imgi):
            img=resize(imgi,INPUT_SIZE)
            min = img.min()
            max = img.max()
            x = 2.0 * (img - min) / (max - min) - 1.0
            return x
        
        all_images= []
        all_labels= []
        PATH=DATA_NAMES[NAME]
        if NAME=='bigearthnet_all':
            #TODO: COMPLETE
            ds= tfds.load('bigearthnet/all',data_dir=DATA_DIR,split='train')
            ds= tfds.as_numpy(ds)
            #bands_10m = tf.stack([ds.B04, ds.B03, ds.B02, ds.B08], axis=3)
            #bands_20m = tf.stack([ds.B05, ds.B06, ds.B07, ds.B8A, ds.B11, ds.B12], axis=3)
            #bands_60m = tf.stack([ds.B01, ds.B09], axis=3)
            #img = tf.concat(
            #[
             #   bands_10m , 
              #  tf.image.resize_bicubic(
               #     bands_20m, 
                #    [120, 120]
                #)
            #], 
            #axis=3
        #)
         #   multi_hot_label = tf.placeholder(tf.float32, shape=(None,43))

        if NAME=='brazildam':
            INPUT_SIZE=(384,384,13)
            LABELS = os.listdir(DATA_NAMES[NAME])
            def createddt(img):
                all_images.append(normalize(tiff.imread(os.path.join(PATH, l, img))))
                all_labels.append(l)
            for l in LABELS:
                Parallel(n_jobs=-1, verbose=1, backend="threading")(map(delayed(createddt), os.listdir(PATH+'/'+l)))
            all_images=np.reshape(all_images,(len(all_images),)+INPUT_SIZE)
            all_labels=np.array(all_labels)
            X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(all_images, all_labels,test_size=SPLIT)
            return X_train, X_test, y_train, y_test

        if NAME=='coffee_scenes':
            X_train=[]
            y_train=[]
            X_test=[]
            y_test=[]
            
            def create_img(img,i,var):
                var.append(normalize(np.asarray(Image.open(os.path.join(PATH,'fold'+str(i), img)))))

            for i in range(1,6):
                labels=pd.read_csv(PATH+'fold'+str(i)+'.txt',sep=".",header=None)
                if i==5:
                    y_test.extend(labels[0][:])
                    Parallel(n_jobs=-1, verbose=1, backend="threading")(delayed(create_img)(img=im,i=i,var=X_test) for im in os.listdir(PATH+'fold'+str(i)))
                else:
                    y_train.extend(labels[0][:])
                    Parallel(n_jobs=-1, verbose=1, backend="threading")(delayed(create_img)(img=im,i=i,var=X_train) for im in os.listdir(PATH+'fold'+str(i)))
            return np.asarray(X_train),np.asarray(X_test),np.asarray(y_train),np.asarray(y_test)
        
        if NAME=='savana':
            LABELS = os.listdir(DATA_NAMES[NAME]+'folds')
            IMAGES = DATA_NAMES[NAME]+'images/'
            all_X=[]
            all_y=[]
            X_train=[]
            X_test=[]
            y_train=[]
            y_test=[]
            def create_img(img,var):
                var.append(normalize(tiff.imread(os.path.join(IMAGES, img+'.tif'))))
            for l in LABELS:
                r=pd.read_csv(DATA_NAMES[NAME]+'folds/'+l,sep=".",header=None)
                all_X.append(r[1][:])
                all_y.append(r[0][:])
            for i in range(5):
                if i==4:
                    y_test=np.asarray(all_y[i])
                    Parallel(n_jobs=-1, verbose=1, backend="threading")(delayed(create_img)(img=im,var=X_test) for im in all_X[i])
                else:
                    y_train.extend(all_y[i])
                    Parallel(n_jobs=-1, verbose=1, backend="threading")(delayed(create_img)(img=im,var=X_train) for im in all_X[i])
            X_train=np.asarray(X_train)
            y_train=np.asarray(y_train)
            X_test=np.asarray(X_test)
            return X_train, X_test, y_train, y_test
