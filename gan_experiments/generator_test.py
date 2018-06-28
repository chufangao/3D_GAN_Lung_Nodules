import keras
import matplotlib.pyplot as plt
import numpy as np

model = keras.models.load_model('gan_generator_3.h5')

model.compile(optimizer=keras.optimizers.Adam(), loss='binary_crossentropy')

noise = np.random.normal(0, 1, (50, 100))
img = model.predict(noise)

gen_imgs = 0.5 * img + 0.5
r = 5
c = 10
fig, axs = plt.subplots(r, c)
cnt = 0
for i in range(r):
    for j in range(c):
        axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
        axs[i, j].axis('off')
        cnt += 1
fig.savefig("images/test_images.png")
plt.close()
