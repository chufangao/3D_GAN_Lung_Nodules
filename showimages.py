import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import pickle

with open('images/generated_nodules100_.pickle','rb') as f:
   posdat = pickle.load(f)

for i in range(posdat.shape[3]):
    print(np.max(posdat[i]))
    plt.imsave('test'+str(i)+'.png',posdat[0,:,:,i,0])
