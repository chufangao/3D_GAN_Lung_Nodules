# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 10:01:50 2017

@author: SMAHESH
"""

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

start_time = time.time()
print("--- %s seconds ---" % (time.time() - start_time))

modelfile = '4.2weights.17-0.04.hdf5'
modelx = keras.models.load_model(modelfile)

#setup
savepath = '/home/cc/Data/'
Xsize = 40
Ysize = 40
Zsize = 18

XYstride = 20
Zstride = 9


thresholds = [x/100 for x in list(range(1, 100))]
thresholds.extend([x/1000 for x in list(range(991, 1000))])
thresholds.extend([x/10000 for x in list(range(9991, 10000))])
FPrates = []
sensitivities = []
fakeSensitivities = []
FPrates = []
sensitivities = []
fakeSensitivities = []

noduleBoxes = None
with open(savepath+"noduleBoxes.pickle", "rb") as f:
    noduleBoxes = pickle.load(f)
fakeNoduleBoxes = None
with open(savepath+"fakeNoduleBoxes.pickle", "rb") as f:
    fakeNoduleBoxes = pickle.load(f)
sliceList = None
with open(savepath+"sliceamount.pickle", "rb") as f:
    sliceList = pickle.load(f)    
valSeries = None
with open(savepath+"workingValidationSeries.pickle", "rb") as f:
    valSeries = pickle.load(f)
numScans = len(valSeries)

sumofFPs = []
sumofTPs = []  
numDetected = [] 
numFakesDetected = [] 
for o in range(len(thresholds)):
    sumofFPs.append(0)
    sumofTPs.append(0)
    numDetected.append(0)
    numFakesDetected.append(0)
k = 0
numNodules = 0
numFakes = 0

Xlow = 0
allboxXs = []
allboxYs = []
Xhigh = Xsize
while Xhigh < 512:
    allboxXs.append([Xlow, Xhigh])
    allboxYs.append([Xlow, Xhigh])
    Xlow += XYstride
    Xhigh += XYstride              
counterx = 0       
for seriesID in valSeries:
    counterx += 1
    print ("File: " + str(counterx))
    inputs = None
    with open(savepath+"ValClipped" + seriesID + ".pickle", 'rb') as f:
        inputs = pickle.load(f)
    inputs = np.array(inputs)
    inputs = inputs.reshape(inputs.shape[0], Xsize, Ysize, Zsize, 1)
    predictions = modelx.predict(inputs, batch_size=48)
 
