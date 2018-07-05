# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 10:33:07 2017

@author: SMAHESH
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 16:59:13 2017

@author: SMAHESH
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 11:50:40 2017

@author: SMAHESH
"""
import random
random.seed(3)
from tensorflow import set_random_seed
set_random_seed(2)


import pickle
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
'''from skimage.util import img_as_ubyte
from skimage.morphology import disk
from skimage.filters import rank'''
import tensorflow as tf
sess = tf.Session()
import numpy as np
from numpy.random import uniform
import keras
from keras.models import Sequential
from keras.initializers import he_normal, Constant
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv3D, MaxPooling3D, BatchNormalization
from keras.optimizers import Adam
from keras import backend as K
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
K.set_session(sess)
import time
start_time = time.time()
print("--- %s seconds ---" % (time.time() - start_time))

# this placeholder will contain our input digits, as flat vectors
x = 40
y = 40
z = 18
grayscale = 1

def return_model(n = 5, drop_rate_conv = 0.09 , drop_rate_FC = 0.56, learn_rate = 0.00024, num_nodes = 64):
    model = Sequential()
    
    #define x, y, z, and  and redo filtersize based on image definitions
    # learn how to do batch normalization
    print ("Running")
    model.add(Conv3D(2**n, 3, strides=(1, 1, 1), activation='relu', input_shape=(x, y, z, grayscale)))
    model.add(BatchNormalization())
    model.add(Conv3D(2**n, 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(drop_rate_conv))
    
    model.add(Conv3D(2**(n+1), 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv3D(2**(n+1), 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(drop_rate_conv))
    
    model.add(Flatten())
    model.add(Dense(num_nodes, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(drop_rate_FC))
    
    #need to redefine output 
    model.add(Dense(2, activation='softmax'))

    adam = Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    #check loss function
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics = ['acc'])

    return model


#dataloading
cutoff = 4606
loadedpos = None
top_threshold = 2446
bottom_threshold = -1434
with open("PositiveAugmented.pickle", "rb") as f:
    print("fine")
    loadedpos = pickle.load(f)
for i in range(len(loadedpos)):
    for j in range(len(loadedpos[i])):
        for k in range(len(loadedpos[i][j])):
            for l in range(len(loadedpos[i][j][k])):
                if loadedpos[i][j][k][l] > top_threshold:
                    loadedpos[i][j][k][l] = top_threshold
                if loadedpos[i][j][k][l] < bottom_threshold:
                    loadedpos[i][j][k][l] = bottom_threshold

smallpos = loadedpos[0:cutoff]
smallpos = np.array(smallpos)
smallpos = smallpos.reshape(smallpos.shape[0], x, y, z, 1)
print (smallpos.shape)
valpos = loadedpos[cutoff:]
valpos = np.array(valpos)
valpos = valpos.reshape(valpos.shape[0], x, y, z, 1)
del loadedpos

loadedneg = None
with open("NegativeAugmented.pickle", "rb") as f:
    print("fine")
    loadedneg = pickle.load(f)
for i in range(len(loadedneg)):
    for j in range(len(loadedneg[i])):
        for k in range(len(loadedneg[i][j])):
            for l in range(len(loadedneg[i][j][k])):
                if loadedneg[i][j][k][l] > top_threshold:
                    loadedneg[i][j][k][l] = top_threshold
                if loadedneg[i][j][k][l] < bottom_threshold:
                    loadedneg[i][j][k][l] = bottom_threshold

smallneg = loadedneg[0:cutoff]
valneg = loadedneg[cutoff:cutoff+819]
del loadedneg
smallneg = np.array(smallneg)
smallneg = smallneg.reshape(smallneg.shape[0], x, y, z, 1)
print (smallneg.shape)
valneg = np.array(valneg)
valneg = valneg.reshape(valneg.shape[0], x, y, z, 1)

x_train = np.concatenate((smallpos, smallneg), axis=0)
'''for place in range(len(x_train)):
    x_train[place] = np.dstack(x_train[place])'''
print ("FinsetupTrain")
print (x_train.shape)
x_train = np.array(x_train)
print (x_train.shape)
'''for k in range(len(x_train)):
    for a in range(len(x_train[k])):
        for b in range(len(x_train[k][a])):
            x_train[k][a][b] = np.asarray(x_train[k][a][b])
        x_train[k][a] = np.asarray(x_train[k][a])
    x_train[k] = np.asarray(x_train[k]) '''
#x_train = np.asarray(x_train)
#print (x_train.shape)    

#print (x_train[0])
print (len(x_train[0][0]))
print (len(x_train[0][0][0]))
y_train = []
#print (x_train)
for i in range(len(smallpos)):
    y_train.append([1,0])
    
for i in range(len(smallneg)):
    y_train.append([0,1])
y_train = np.array(y_train)     


x_test = np.concatenate((valpos, valneg), axis=0)
'''for place in range(len(x_test)):
    x_test[place] = np.dstack(x_test[place])'''
print ("FinsetupTest")
x_test = np.array(x_test)
y_test = []
#print (x_test)
for i in range(len(valpos)):
    y_test.append([1,0])
    
for i in range(len(valneg)):
    y_test.append([0,1])
y_test = np.array(y_test)     

tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)     
modelcheck = keras.callbacks.ModelCheckpoint('4.2weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

modelx = return_model()

history = modelx.fit(x_train, y_train, batch_size=60, epochs=50, callbacks=[tbCallBack, modelcheck], validation_data=[x_test, y_test])
#score = model.evaluate(x_test, y_test, batch_size=32)

def generate_results(y_test, y_score, filename):
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.savefig('ROC' + filename + '.png')
    print('AUC: %f' % roc_auc)
    with open('AUC' + filename + '.txt', 'w') as f:
        f.write('ROC CURVE: %f' % roc_auc)
    plt.clf()

y_score = modelx.predict(x_test)
generate_results(y_test[:, 0], y_score[:, 0], '13I')

#generate_results(y_test[:, 0], y_score[:, 0], 'ROC13R.png')

print (history.history)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('accplot13I.png')
plt.clf()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('lossplot13I.png')



# model.save(filepath) #to save NN model
# keras.models.load_model(filepath) #to load model  

'''If you need to save the weights of a model, you can do so in HDF5 with the code below.

Note that you will first need to install HDF5 and the Python library h5py, which do not come bundled with Keras.

model.save_weights(filename.h5)
model.load_weights(filename.h5) '''

''''You can do batch training using model.train_on_batch(x, y) and model.test_on_batch(x, y). See the models documentation.

Alternatively, you can write a generator that yields batches of training data and use the method  model.fit_generator(data_generator, steps_per_epoch, epochs).

You can see batch training in action in our CIFAR10 example.'''