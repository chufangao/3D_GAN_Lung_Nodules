import keras
import matplotlib.pyplot as plt
import numpy as np

discriminator = keras.models.load_model('untrained_gan_classifier.h5')
discriminator.summary()

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
X_train = x_train / 127.5 - 1.
X_train = np.expand_dims(X_train, axis=3)