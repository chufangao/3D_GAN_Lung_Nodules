import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

experimentPath = 'C:/Users/CGAO8/Documents/Code/experiment1/'
experimentFiles = os.listdir(experimentPath)

for file in experimentFiles:
    trialPath = experimentPath+file+'/'

    plt.plot(pickle.load(open(trialPath+'aug_FPratesAdj1.pickle', 'rb')), pickle.load(open(trialPath+'aug_sensitivities1.pickle', 'rb')))

plt.xlabel('Adjusted FPS per scan')
plt.ylabel('Sensitivity')

print(experimentFiles)
plt.legend(experimentFiles)
# plt.savefig('compared.png')
plt.show()
