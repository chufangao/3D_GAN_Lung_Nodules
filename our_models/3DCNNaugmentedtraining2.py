'''
trains and saves models based on experiments specified in paper
'''

import random
random.seed(3)
from tensorflow import set_random_seed
set_random_seed(2)


import pickle
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
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
import os
start_time = time.time()
print("--- %s seconds ---" % (time.time() - start_time))

# this placeholder will contain our input digits, as flat vectors
x = 40
y = 40
z = 18
grayscale = 1

#the name of the experiment; used to create a directory to store results
experiment_name = 'experimental_experiment1'

EPOCHS = 20
#the size of the noise vector
latent_dim = 200

#the number of fake positive examples and real negative examples to add to the base data set for each trial
experiment_trials = [[0,0], [.1,0], [1.0,0], [2.0,0], [.1,.1], [1.0,1.0], [2.0,2.0]]

#these examples are taken equally from both the positive and negative examples
validation_percentage = .2

#true if a model should be trained without augmented data
process_control_groups = False
# how many examples we want to generate at a time
generate_quantity = 500
# how many times we want to do it
augmentation_iterations = 2
#the file containing the model for generating new training data
generator_file = 'saved_models/g1.h5'

#the number of negative examples to add at a time
negative_quantity = generate_quantity
#the number of times to add negative examples
negative_iterations = augmentation_iterations

#the directory for saving models
target_directory = 'saved_models/'

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
    plt.savefig(filename+'ROC.png')
    print('AUC: %f' % roc_auc)
    with open(filename + 'AUC.txt', 'w') as f:
        f.write('ROC CURVE: %f' % roc_auc)
    plt.clf()

#use a file to a instantiate a model for generating new training examples
def load_generator():
    g_model = keras.models.load_model(generator_file)
    return g_model

def denormalize_img(normalized_image):
    rval = normalized_image
    rval *= 1940
    rval += 506
    return rval


# dataloading
loadedpos = None
#are the thresholds still used?
top_threshold = 2446
bottom_threshold = -1434
with open("/home/cc/Data/PositiveAugmented.pickle", "rb") as f:
    print("loaded pos data")
    loadedpos = pickle.load(f)

cutoff = int(validation_percentage*len(loadedpos))
valpos = loadedpos[0:cutoff]
valpos = np.array(valpos)
valpos = valpos.reshape(valpos.shape[0], x, y, z, 1)
smallpos = loadedpos[cutoff:]
smallpos = np.array(smallpos)
smallpos = smallpos.reshape(smallpos.shape[0], x, y, z, 1)
print ('pos data shape ', smallpos.shape, ' val pos shape', valpos.shape)
del loadedpos
pos_len = len(smallpos)

loadedneg = None
with open("/home/cc/Data/NegativeAugmented.pickle", "rb") as f:
    print("fine")
    loadedneg = pickle.load(f)
valneg = loadedneg[0:cutoff]
neg_cutoff = cutoff + pos_len
smallneg = loadedneg[cutoff: neg_cutoff]
# loadedneg is currently necessary for fetching new negative examples
del loadedneg
smallneg = np.array(smallneg)
smallneg = smallneg.reshape(smallneg.shape[0], x, y, z, 1)
valneg = np.array(valneg)
valneg = valneg.reshape(valneg.shape[0], x, y, z, 1)
print ('neg data shape', smallneg.shape, 'val neg shape', valneg.shape)

#this set contains the base training data for testing on
base_set = np.concatenate((smallpos, smallneg), axis=0)
print ("Train data shape", base_set.shape)
test_data = np.concatenate((valpos, valneg), axis=0)
print ("Val data shape", test_data.shape)

# create labels for training data
base_label = []
for i in range(len(smallpos)):
    base_label.append([1, 0])
for i in range(len(smallneg)):
    base_label.append([0, 1])
base_label = np.array(base_label)

# create labels for testing data
test_label = []
for i in range(len(valpos)):
    test_label.append([1, 0])
for i in range(len(valneg)):
    test_label.append([0, 1])
test_label = np.array(test_label)

tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)     
modelcheck = keras.callbacks.ModelCheckpoint('4.2weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

# load generator
example_generator = load_generator()

if not os.path.exists(experiment_name):
    os.mkdir(experiment_name)                                                                                                                                                                            

# perform experiments
for i in experiment_trials:
    print('experiment:', i)
    train_set = base_set
    train_label = base_label

    # generate fake pos data according to experiment
    if i[0] != 0:
        fake_pos_quantity = int(i[0] * pos_len)
        the_noise = np.random.normal(0, 1, (fake_pos_quantity, latent_dim))
        fake_pos_data = example_generator.predict(the_noise)
        fake_pos_data = denormalize_img(fake_pos_data)
        fake_pos_label = []
        for j in range(len(fake_pos_data)):
            fake_pos_label.append([1, 0])
        #print('train_data type:' + str(type(base_set)) + '; new_train_data type: ' + str(type(fake_pos_data)))
        train_set = np.concatenate((base_set, fake_pos_data), 0)
        train_label = np.concatenate((base_label, fake_pos_label))
        # print('train set shape', train_set.shape, 'train label shape', train_label.shape)

    # get neg data according to experiment
    if i[1] != 0:
        new_neg_quantity = int(i[1] * pos_len)
        new_neg_data = loadedneg[neg_cutoff : neg_cutoff + pos_len]
        new_neg_label = []
        for j in range(len(new_neg_data)):
            new_neg_label.append([0, 1])
        new_neg_data = np.array(new_neg_data)
        new_neg_data = new_neg_data.reshape(new_neg_data.shape[0], x, y, z, 1)
        #print(train_set.shape, new_neg_data.shape)
        train_set = np.concatenate((train_set, new_neg_data), 0)
        #print(train_label.shape, new_neg_label.shape)
        train_label = np.concatenate((train_label, new_neg_label), 0)

    print('train set shape', train_set.shape, 'train label shape', train_label.shape)

    modelx = return_model()
    history = modelx.fit(train_set, train_label,
            batch_size=60, epochs=EPOCHS,
            #callbacks=[tbCallBack, modelcheck],
            validation_data=[test_data, test_label])
    modelx.save(target_directory+'classifier_model_'+str(i[0])+':'+str(i[1])+'-aug.h5')
    # score = model.evaluate(x_test, y_test, batch_size=32)

    y_score = modelx.predict(test_data)
    generate_results(test_label[:, 0], y_score[:, 0], target_directory+'13I'+str(i[0])+':'+str(i[1]))


#print (history.history)
#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.savefig('accplot13I.png')
#plt.clf()
# summarize history for loss
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.savefig('lossplot13I.png')
