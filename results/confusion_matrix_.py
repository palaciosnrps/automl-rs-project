#This code prints the model accuracy and model parameters configuration of the saved models (best model obtained during experiments).
#It also creates a pdf file with the confusion matrix of the results on the test set.

import sys
import time
from joblib import Parallel, delayed
import tensorflow as tf
import tensorflow_datasets as tfds
import sklearn.model_selection
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras import optimizers
from tensorflow.keras.layers import  Activation, AveragePooling2D, BatchNormalization, Conv2D, Convolution2D, Dense, Dropout, MaxPooling2D, MaxPool2D, Flatten, Input, GlobalMaxPooling2D, ZeroPadding2D
import autokeras as ak
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from datasets import create_dataset

dataname=sys.argv[1] 
model_path=sys.argv[2]
plot_name=sys.argv[3]
labelsi=sys.argv[4]
X1,X2,y1,y2=create_dataset(dataname)

model = keras.models.load_model(model_path)
test_loss, test_acc = model.evaluate(X2,y2, verbose=2)
print("Accuracy mod_auto:",test_acc)
print(model.get_config())

pred=model.predict(X2)
y_pred = np.argmax(pred, axis=1)

###This function is a modification of the one founded in 
#https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title= 'Normalized confusion matrix',
                          cmap=plt.cm.Blues):

    # Compute confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax,fraction=0.046, pad=0.04)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    # Rotate the tick labels and set.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
    for i in range(cm.shape[0]):
    	for j in range(cm.shape[1]):
    		text=ax.text(j,i,'{:03.2f}'.format(cm[i,j]),ha='center',va='center')
    return ax

labels=labelsi
plot_confusion_matrix(y2,y_pred,classes=labels, normalize=True, title='Normalized confusion matrix')
plt.savefig(plot_name)
