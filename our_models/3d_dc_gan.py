from keras.datasets import cifar10
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pickle

#these represent the dimensions of the medical images
x = 40
y = 40
z = 18
grayscale = 1

def load_augmented_positive(cutoff=1000):
    loadedpos = None
    with open('PositiveAugmented.pickle', 'rb') as f:
        loadedpos = pickle.load(f)

    #smallpos = loadedpos[0:cutoff]
    smallpos = np.array(smallpos)
    smallpos = smallpos.reshape(smallpos.shape[0], x, y, z, grayscale)
    print('Positive nodules shape: '+smallpos.shape)
    # valpos = loadedpos[cutoff:]
    # valpos = np.array(valpos)
    # valpos = valpos.reshape(valpos.shape[0], x, y, z, grayscale)
    return

def load_augmented_positive(cutoff=1000):
    loadedneg = None
    with open('NegativeAugmented.pickle', 'rb') as f:
        loadedneg = pickle.load(f)

    smallneg = loadedneg[0:cutoff]
    valneg = loadedneg[cutoff:cutoff + 819]
    del loadedneg
    smallneg = np.array(smallneg)
    smallneg = smallneg.reshape(smallneg.shape[0], x, y, z, 1)
    print(smallneg.shape)
    valneg = np.array(valneg)
    valneg = valneg.reshape(valneg.shape[0], x, y, z, 1)
    return

load_augmented_positive()
