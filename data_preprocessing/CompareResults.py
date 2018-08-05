import pickle
import numpy as np
import matplotlib.pyplot as plt
import os


def graphOne(experimentPath, labels):
    for file in os.listdir(experimentPath)[1:]:
        trialPath = experimentPath + file + '/'
        try:
            plt.plot(pickle.load(open(trialPath + 'aug_FPratesAdj1.pickle', 'rb')), pickle.load(open(trialPath + 'aug_sensitivities1.pickle', 'rb')))
        except:
            pass

    plt.xlabel('Adjusted False Positives per Scan')
    plt.ylabel('Sensitivity')
    plt.legend(labels)
    plt.show()

def graphManyVsMany(experimentPath, controlsPath, labels):
    totalFPrate = []
    totalSens = []
    controlFPRate = []
    controlSens = []

    for file in os.listdir(experimentPath)[1:]:
        trialPath = experimentPath + file + '/'
        try:
            totalFPrate.extend(pickle.load(open(trialPath + 'aug_FPratesAdj1.pickle', 'rb')))
            totalSens.extend(pickle.load(open(trialPath + 'aug_sensitivities1.pickle', 'rb')))
        except:
            pass

    for file in os.listdir(controlsPath):
        trialPath = controlsPath + file + '/'
        try:
            controlFPRate.extend(pickle.load(open(trialPath + 'aug_FPratesAdj1.pickle', 'rb')))
            controlSens.extend(pickle.load(open(trialPath + 'aug_sensitivities1.pickle', 'rb')))
        except:
            pass

    print(len(totalFPrate), len(totalSens))
    plt.scatter(totalFPrate, totalSens, marker='.')
    plt.plot(np.linspace(0, 500, 100), np.poly1d(np.polyfit(totalFPrate, totalSens, 4))(np.linspace(0, 500, 100)), 'g')

    print(len(controlFPRate), len(controlSens))
    plt.scatter(controlFPRate, controlSens, marker='.')
    plt.plot(np.linspace(0, 500, 100), np.poly1d(np.polyfit(controlFPRate, controlSens, 4))(np.linspace(0, 500, 100)), 'r--')

    plt.xlabel('Adjusted False Positives per Scan')
    plt.ylabel('Sensitivity')
    plt.legend(labels)
    plt.show()

if __name__ == '__main__':
    experimentPath = 'C:/Users/gaoan/Documents/Python Scripts/deep_learning_reu/images/experiment4/trials/'
    controlsPath = 'C:/Users/gaoan/Documents/Python Scripts/deep_learning_reu/images/controls/trials/'
    labels = []
    labels.append('1 aug 1 neg')
    labels.append('control')
    labels.append('test scatterplot')
    labels.append('control scatterplot')
    # graphManyVsMany(experimentPath, controlsPath, labels)
    graphOne(experimentPath, os.listdir(experimentPath))