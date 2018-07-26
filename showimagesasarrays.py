import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle

posdat = pickle.load(open('images/gen_nod1.pickle','rb'))
# posdat = pickle.load(open('images/PositiveAugmented.pickle','rb'))

threshold = .9
# print(np.average(posdat, axis=(1,2,3,4))); exit()
print('white imgs', [i for i, v in enumerate(posdat) if np.average(v) > threshold])
posdat = np.array(posdat)
print(posdat.shape)
fig, axs = plt.subplots(3,6)

cnt = 0
for k in range(0,16):
    for i in range(3):
        for j in range(6):
            axs[i,j].imshow(0.5 * posdat[k,:,:,cnt,0] + 0.5, cmap='gray')
            # axs[i, j].imshow(0.5 * posdat[k, :, :, cnt] + 0.5, cmap='gray')
            axs[i,j].axis('off')
            cnt += 1

    cnt = 0
    fig.savefig('test'+str(k)+'.png')
