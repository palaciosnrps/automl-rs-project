import time
from joblib import Parallel, delayed
import tensorflow as tf
import tensorflow_datasets as tfds
import sklearn.model_selection
from tensorflow.keras.utils import to_categorical
import numpy as np

DATA_DIR='/data/s2105713/data'
euro_rgb= tfds.load('eurosat/rgb', data_dir=DATA_DIR,split='train', as_supervised=True,shuffle_files=True)
euro_all= tfds.load('eurosat/all', data_dir=DATA_DIR,split='train', as_supervised=True,shuffle_files=True)
big_rgb= tfds.load('bigearthnet/rgb', data_dir=DATA_DIR,split='train')
big_all= tfds.load('bigearthnet/all', data_dir=DATA_DIR,split='train')
