"""An implementation of the improved WGAN described in https://arxiv.org/abs/1704.00028

The improved WGAN has a term in the loss function which penalizes the network if its gradient
norm moves away from 1. This is included because the Earth Mover (EM) distance used in WGANs is only easy
to calculate for 1-Lipschitz functions (i.e. functions where the gradient norm has a constant upper bound of 1).

The original WGAN paper enforced this by clipping weights to very small values [-0.01, 0.01]. However, this
drastically reduced network capacity. Penalizing the gradient norm is more natural, but this requires
second-order gradients. These are not supported for some tensorflow ops (particularly MaxPool and AveragePool)
in the current release (1.0.x), but they are supported in the current nightly builds (1.1.0-rc1 and higher).

To avoid this, this model uses strided convolutions instead of Average/Maxpooling for downsampling. If you wish to use
pooling operations in your discriminator, please ensure you update Tensorflow to 1.1.0-rc1 or higher. I haven't
tested this with Theano at all.

The model saves images using pillow. If you don't have pillow, either install it or remove the calls to generate_images.
"""
import argparse
import os
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers.merge import _Merge
from keras.layers.convolutional import Convolution3D, UpSampling3D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.datasets import mnist
from keras import backend as K
from functools import partial

try:
    from PIL import Image
except ImportError:
    print('This script depends on pillow! Please install it (e.g. with pip install pillow)')
    exit()

BATCH_SIZE = 64
TRAINING_RATIO = 5  # The training ratio is the number of discriminator updates per generator update. The paper uses 5.
GRADIENT_PENALTY_WEIGHT = 10  # As per the paper

def make_generator():
    """Creates a generator model that takes a 100-dimensional noise vector as a "seed", and outputs images
    of size 28x28x1."""
    model = Sequential()
    model.add(Dense(1024, input_dim=100))
    model.add(LeakyReLU())
    model.add(Dense(128 * 10 * 10 * 9))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    if K.image_data_format() == 'channels_first':
        model.add(Reshape((128, 10, 10, 9), input_shape=(128 * 10 * 10 * 9,)))
        bn_axis = 1
    else:
        model.add(Reshape((10, 10, 9, 128), input_shape=(128 * 10 * 10 * 9,)))
        bn_axis = -1
    model.add(UpSampling3D(size=(2, 2, 2)))
    model.add(BatchNormalization(axis=bn_axis))
    model.add(LeakyReLU())
    model.add(Convolution3D(64, (3, 3, 3), padding='same'))
    model.add(BatchNormalization(axis=bn_axis))
    model.add(LeakyReLU())
    model.add(UpSampling3D(size=(2, 2, 1)))
    model.add(BatchNormalization(axis=bn_axis))
    model.add(LeakyReLU())
    # Because we normalized training inputs to lie in the range [-1, 1],
    # the tanh function should be used for the output of the generator to ensure its output
    # also lies in this range.
    model.add(Convolution3D(1, (3, 3, 3), padding='same', activation='tanh'))
    return model

def tile_images(image_stack):
    """Given a stacked tensor of images, reshapes them into a horizontal tiling for display."""
    assert len(image_stack.shape) == 3
    image_list = [image_stack[i, :, :] for i in range(image_stack.shape[0])]
    tiled_images = np.concatenate(image_list, axis=1)
    return tiled_images

def generate_images(generator_model, output_dir, epoch):
    """Feeds random seeds into the generator and tiles and saves the output to a PNG file."""
    test_image_stack = generator_model.predict(np.random.rand(10, 100))
    test_image_stack = (test_image_stack * 127.5) + 127.5
    test_image_stack = np.squeeze(np.round(test_image_stack).astype(np.uint8))
    tiled_output = tile_images(test_image_stack)
    tiled_output = Image.fromarray(tiled_output, mode='L')  # L specifies greyscale
    outfile = os.path.join(output_dir, 'epoch_{}.png'.format(epoch))
    tiled_output.save(outfile)

gen = make_generator()
input = Input(shape=(100,))
layers = gen(input)
model = Model(inputs = [input], outputs = [layers])
model.compile(optimizer=Adam(), loss=lambda y_true, y_pred: K.mean(y_true * y_pred))
model.layers[1].summary()