import pickle
import numpy as np
import os

bottom_threshold = -1434
top_threshold = 2446
handle = '/home/cc/Data/NegativeAugmented.pickle'
path = '/home/cc/Data/'

def processValidation(path):
    for handle in [i for i in os.listdir(path) if 'ValClipped' in i]:
        d = pickle.load(open(path+handle, 'rb'))
        x_train = np.array(d)
        x_train = np.clip(x_train, bottom_threshold, top_threshold)
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], x_train.shape[3], 1))
        minx = np.amin(x_train)
        halfRange = (np.amax(x_train) - minx) / 2.0
        if minx < 0:
            x_train -= minx
        x_train = (x_train - halfRange) / halfRange
        pickle.dump(x_train.astype('float32'), open(path+handle,'wb'), protocol=pickle.HIGHEST_PROTOCOL)

def processAug(path):
    listdata = []
    d = pickle.load(open(path, 'rb'))
    x_train = np.array(d)
    x_train = np.clip(x_train, bottom_threshold, top_threshold)
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], x_train.shape[3], 1))
    minx = np.amin(x_train)
    halfRange = (np.amax(x_train) - minx) / 2.0
    if minx < 0:
        x_train -= minx
    x_train = (x_train - halfRange) / halfRange
    pickle.dump(x_train.astype('float32'), open(path,'wb'), protocol=pickle.HIGHEST_PROTOCOL)


processAug('/home/cc/Data/NegativeAugmented.pickle')

