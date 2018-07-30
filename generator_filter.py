import keras
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from skimage import feature

latent_dim = 200

#use a file to a instantiate a model for generating new training examples
def load_generator(generator_file):
    g_model = keras.models.load_model(generator_file)
    return g_model

def generate_images(generator, quantity):
    the_noise = np.random.normal(0, 1, (quantity, latent_dim))
    return generator.predict(the_noise)

def filter_images(images):
    filtered = list()
    unfiltered = list()
    for img in images:
        cannyslices = list()
        for i in range(18):
            slice = img[:, :, i, 0]
            #print(slice.shape)
            cannyslice = feature.canny(slice, sigma = 3)
            cannyslices.append(cannyslice)
        cannyimg = np.stack(cannyslices, 2)
        cannyimg = np.reshape(cannyimg, (40,40,18,1))
        if np.mean(img) > .9:
            #filtered.append(cannyimg)
            filtered.append(img)
        else:
            #unfiltered.append(cannyimg)
            unfiltered.append(img)

    return filtered, unfiltered


generator_file = 'our_models/saved_models/g1.h5'

generator = load_generator(generator_file)
fake_images = generate_images(generator, 50)
filtered, unfiltered = filter_images(fake_images)

fig, axs = plt.subplots(3, 6)

filtered_dir = 'images/filtered/'
unfiltered_dir = 'images/unfiltered/'

cnt = 0
k = 1
for img in filtered:
    for i in range(3):
        for j in range(6):
            axs[i,j].imshow(0.5 * img[:, :, cnt, 0] + 0.5, cmap='gray')
            axs[i,j].axis('off')
            cnt += 1

    cnt = 0
    fig.savefig(filtered_dir + 'test'+str(k)+'.png')
    k += 1

for img in unfiltered:
    for i in range(3):
        for j in range(6):
            axs[i, j].imshow(0.5 * img[:, :, cnt, 0] + 0.5, cmap='gray')
            axs[i, j].axis('off')
            cnt += 1

    cnt = 0
    fig.savefig(unfiltered_dir + 'test' + str(k) + '.png')
    k += 1

print('done')

