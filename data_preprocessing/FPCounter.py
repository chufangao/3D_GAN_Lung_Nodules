# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 13:10:16 2017

@author: SMAHESH
"""

#creates sliding-box inputs for one scan, separates into positive and negative, evaluates with model, etc

from sklearn.model_selection import RandomizedSearchCV
import tensorflow as tf
sess = tf.Session()

from numpy.random import uniform
import keras
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv3D, MaxPooling3D, BatchNormalization
from keras.optimizers import Adam
from keras import backend as K
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
K.set_session(sess)
import time
start_time = time.time()
print("--- %s seconds ---" % (time.time() - start_time))

import pickle
import gc
import pandas as pd
from collections import OrderedDict
import numpy as np
import random
random.seed(10)

#replace weight file with wieghts you want to use
modelfile = '4.3aweights.33-0.05.hdf5'
modelx = keras.models.load_model(modelfile)

Xsize = 40
Ysize = 40
Zsize = 18

XYstride = 20
Zstride = 9

noduleBoxes = None
with open("noduleBoxesJP.pickle", "rb") as f:
    noduleBoxes = pickle.load(f)
    print (noduleBoxes.keys())
fakeNoduleBoxes = None
with open("fakeNoduleBoxesJP.pickle", "rb") as f:
    fakeNoduleBoxes = pickle.load(f)
with open("sliceamountJP.pickle", "rb") as f:
    sliceList = pickle.load(f)    
valSeries = None
with open("workingValidationSeries.pickle", "rb") as f:
    valSeries = pickle.load(f)
numScans = len(valSeries)
inputs =  None
#replace string with file you want to check
with open('TRAINInputs1.3.6.1.4.1.14519.5.2.1.6279.6001.245546033414728092794968890929.pickle', 'rb') as handle:
        inputs = pickle.load(handle)

#creating coordinate boxes
print (len(inputs))

#these should match, if they don't we messed up
print("Should match:")
#print(len(coords))
print(len(inputs))

alpha = []
alphacoords = []

#nodules = noduleBoxes[filestring]
#fakeNodules = fakeNoduleBoxes[filestring]
for i in range(len(inputs)):
    alpha.append(inputs[i])
print(len(alpha))

upperpred = None
negsamples = None
first = True
for a in range (1):
    alpha = np.array(alpha) 
    #alphacoords = np.array(alphacoords)
    alpha = alpha.reshape(alpha.shape[0], Xsize, Ysize, Zsize, 1)
    ypred = modelx.predict(alpha, batch_size=30)
    #ypred = np.delete(arr, np.s_[1::1], 1) #gets the prob its a true positive
    indexes = []
    valuelist = []
    for i in range(len(ypred)):
        if ypred[i][0] >= .5:
            indexes.append(i)
            valuelist.append(ypred[i][0])
    print(len(indexes))
    alpha = np.take(alpha, indexes, axis = 0)
    #alphacoords = np.take(alphacoords, indexes, axis = 0)
    valuelist = np.array(valuelist)
    if first:
        upperpred = valuelist
        negsamples = alpha
        #FPcoords = alphacoords
        first = False
    else:
        upperpred = np.concatenate((upperpred, valuelist), axis = 0)
        negsamples = np.concatenate((negsamples, alpha), axis = 0)
        #FPcoords = np.concatenate((FPcoords, alphacoords), axis = 0)
    del valuelist
    del ypred
    gc.collect()       
    
with open("Negsamples.pickle", 'wb') as handle:
    pickle.dump(negsamples, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open("PredictedValues.pickle", 'wb') as handle:
    pickle.dump(upperpred, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

print("FPs:")
print(len(negsamples))


