import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

experimentPath = 'C:/Users/CGAO8/Documents/Deep Learning Practice/images/experiment4/trials/'
experimentFiles = os.listdir(experimentPath)
thresholds = [x/100 for x in list(range(1, 100))]
thresholds.extend([x/1000 for x in list(range(991, 1000))])
thresholds.extend([x/10000 for x in list(range(9991, 10000))])

for file in experimentFiles:
    trialPath = experimentPath+file+'/'
    # tmp = pickle.load(open(trialPath+'aug_fakeSensitivities1.pickle', 'rb'))
    # print(len(tmp))
    plt.plot(pickle.load(open(trialPath+'aug_FPratesAdj1.pickle', 'rb')), pickle.load(open(trialPath+'aug_sensitivities1.pickle', 'rb')))

plt.xlabel('Adjusted FPS per scan')
plt.ylabel('Sensitivity')

print(experimentFiles)
plt.legend(experimentFiles)
# plt.savefig('compared.png')
plt.show()
