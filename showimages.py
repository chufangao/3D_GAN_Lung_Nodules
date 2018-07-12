import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import pickle

EPOCH = '300'

with open('C:\Users\sclark54\Desktop\generated_nodules'+EPOCH+'.pickle','rb') as f:
   posdat = pickle.load(f)

dir = 'C:/Users/sclark54/Desktop/pics/epoch'+EPOCH+'/'
for i in range(posdat.shape[3]):
    print(np.max(posdat[i]))
    plt.imsave('nodule'+EPOCH+'_'+str(i)+'.png',posdat[0,:,:,i,0])
