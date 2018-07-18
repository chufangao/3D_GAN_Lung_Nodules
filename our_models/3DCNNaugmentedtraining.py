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
from keras import backend as K, Input, Model
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

#true if a model should be trained without augmented data
control_group = False
# how many examples we want to generate at a time
generate_quantity = 500
# how many times we want to do it
augmentation_iterations = 3
#the file containing the model for generating new training data
generator_file = 'g_model100.h5'



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

#use a file to a instantiate a model for generating new training examples
def load_generator():
    g_model = keras.models.load_model(generator_file)
    g_input = Input(shape=(100,))
    g_layers = g_model(g_input)
    g_compiled = Model.compile(inputs=[g_input], outputs=[g_layers])
    return g_compiled

#dataloading
cutoff = 4606
loadedpos = None
top_threshold = 2446
bottom_threshold = -1434
with open("/home/cc/Data/PositiveAugmented.pickle", "rb") as f:
    print("fine")
    loadedpos = pickle.load(f)

smallpos = loadedpos[0:cutoff]
smallpos = np.array(smallpos)
smallpos = smallpos.reshape(smallpos.shape[0], x, y, z, 1)
print (smallpos.shape)
valpos = loadedpos[cutoff:]
valpos = np.array(valpos)
valpos = valpos.reshape(valpos.shape[0], x, y, z, 1)
del loadedpos

loadedneg = None
with open("/home/cc/Data/NegativeAugmented.pickle", "rb") as f:
    print("fine")
    loadedneg = pickle.load(f)

smallneg = loadedneg[0:cutoff]
valneg = loadedneg[cutoff:cutoff+819]
del loadedneg
smallneg = np.array(smallneg)
smallneg = smallneg.reshape(smallneg.shape[0], x, y, z, 1)
print (smallneg.shape)
valneg = np.array(valneg)
valneg = valneg.reshape(valneg.shape[0], x, y, z, 1)

train_data = np.concatenate((smallpos, smallneg), axis=0)
'''for place in range(len(x_train)):
    x_train[place] = np.dstack(x_train[place])'''
print ("FinsetupTrain")
print (train_data.shape)
train_data = np.array(train_data)
print (train_data.shape)
'''for k in range(len(x_train)):
    for a in range(len(x_train[k])):
        for b in range(len(x_train[k][a])):
            x_train[k][a][b] = np.asarray(x_train[k][a][b])
        x_train[k][a] = np.asarray(x_train[k][a])
    x_train[k] = np.asarray(x_train[k]) '''
#x_train = np.asarray(x_train)
#print (x_train.shape)    

#print (x_train[0])
print (len(train_data[0][0]))
print (len(train_data[0][0][0]))
train_label = []
#print (x_train)
for i in range(len(smallpos)):
    train_label.append([1, 0])
    
for i in range(len(smallneg)):
    train_label.append([0, 1])
train_label = np.array(train_label)


test_data = np.concatenate((valpos, valneg), axis=0)
'''for place in range(len(x_test)):
    x_test[place] = np.dstack(x_test[place])'''
print ("FinsetupTest")
test_data = np.array(test_data)
test_label = []
#print (x_test)
for i in range(len(valpos)):
    test_label.append([1, 0])
    
for i in range(len(valneg)):
    test_label.append([0, 1])
test_label = np.array(test_label)

tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)     
modelcheck = keras.callbacks.ModelCheckpoint('4.2weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

if control_group:
    modelx = return_model()

    history = modelx.fit(train_data, train_label, batch_size=60, epochs=20, callbacks=[tbCallBack, modelcheck], validation_data=[test_data, test_label])
    modelx.save('classifier_model_0-aug.h5')
    #score = model.evaluate(x_test, y_test, batch_size=32)

    y_score = modelx.predict(test_data)
    generate_results(test_label[:, 0], y_score[:, 0], '13I_control')

example_generator = load_generator()
for i in range(augmentation_iterations):
    the_noise = np.random.normal(0,1,(100, generate_quantity))
    new_train_data = example_generator.predict(the_noise)
    new_train_label = []
    for j in range(len(new_train_data)):
        new_train_label.append([1,0])
    train_data = np.concatenate(train_data, new_train_data)
    train_label = np.concatenate(train_label, new_train_label)

    modelx = return_model()
    history = modelx.fit(train_data, train_label, batch_size=60, epochs=20, callbacks=[tbCallBack, modelcheck],
                         validation_data=[test_data, test_label])
    modelx.save('classifier_model_'+str((1+i)*generate_quantity)+'-aug.h5')
    # score = model.evaluate(x_test, y_test, batch_size=32)

    y_score = modelx.predict(test_data)
    generate_results(test_label[:, 0], y_score[:, 0], '13I'+str(i+1))


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
