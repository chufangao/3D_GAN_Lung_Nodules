import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

experimentPath = 'C:/Users/CGAO8/Documents/Deep Learning Practice/images/experiment2/'
# experimentFiles = os.listdir(experimentPath)
# thresholds = [x/100 for x in list(range(1, 100))]
# thresholds.extend([x/1000 for x in list(range(991, 1000))])
# thresholds.extend([x/10000 for x in list(range(9991, 10000))])
actualFiles = []


for file in os.listdir(experimentPath):
    trialPath = experimentPath+file+'/'
    try:
        if file == 'control':
            plt.plot(pickle.load(open(trialPath+'aug_FPratesAdj1.pickle', 'rb')), pickle.load(open(trialPath+'aug_sensitivities1.pickle', 'rb')), '--')
        else:
            plt.plot(pickle.load(open(trialPath + 'aug_FPratesAdj1.pickle', 'rb')),
                     pickle.load(open(trialPath + 'aug_sensitivities1.pickle', 'rb')))
        actualFiles.append(file)
    except:
        pass
    # plt.plot(thresholds,
    #          pickle.load(open(trialPath + 'aug_sensitivities1.pickle', 'rb')))

plt.xlabel('Adjusted False Positives per Scan')
plt.ylabel('Sensitivity')

labels = []
for i in actualFiles:
    if i in ['trial_0','trial_1','trial_2','trial_3','trial_5','trial_6','trial_7']:
        labels.append('.3 aug 0 neg')
    elif i == 'control':
        labels.append('control')
    else:
        labels.append('.4 aug .4 neg')
plt.legend(labels)
# plt.savefig('compared.png')
plt.show()
