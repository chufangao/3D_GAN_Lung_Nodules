import keras
import numpy as np
import keras.backend as K
from keras.layers.core import Lambda
import pickle

with open('Data/PositiveAugmented.pickle', 'rb') as f:
    posdat = pickle.load(f)

print(posdat.shape)

def get_feat(x):
    avg = K.mean(x, keepdims=True)
    std = K.std(x, keepdims=True)
    merged = K.concatenate([avg,std])
    return merged

inputs = keras.Input(shape=(40,40,18,1,))
layer_1 = Lambda(get_feat)(inputs)
outputs = keras.layers.Flatten()(layer_1)

model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()
results = model.predict(posdat)
print(results.shape)

