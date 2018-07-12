import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import pickle

with open('generated_nodules50.pickle','rb') as f:
   posdat = pickle.load(f)

for i in range(posdat.shape[3]):
    gen_img = 0.5 * posdat[0,:,:,i,0] + 0.5
    plt.imsave('test'+str(i)+'.png', gen_img, cmap='gray')
