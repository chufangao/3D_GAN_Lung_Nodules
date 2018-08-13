import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

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

    for file in os.listdir(experimentPath)[:10]:
        trialPath = experimentPath + file + '/'
        try:
            totalFPrate.extend(pickle.load(open(trialPath + 'aug_FPratesAdj1.pickle', 'rb')))
            totalSens.extend(pickle.load(open(trialPath + 'aug_sensitivities1.pickle', 'rb')))
        except:
            pass

    for file in os.listdir(controlsPath)[:10]:
        trialPath = controlsPath + file + '/'
        try:
            controlFPRate.extend(pickle.load(open(trialPath + 'aug_FPratesAdj1.pickle', 'rb')))
            controlSens.extend(pickle.load(open(trialPath + 'aug_sensitivities1.pickle', 'rb')))
        except:
            pass

    print(len(totalFPrate), len(totalSens))
    plt.scatter(totalFPrate, totalSens, c='#5ABFFC', marker='.')
    plt.plot(np.linspace(0, 500, 100), np.poly1d(np.polyfit(totalFPrate, totalSens, 4))(np.linspace(0, 500, 100)), 'b')

    print(len(controlFPRate), len(controlSens))
    plt.scatter(controlFPRate, controlSens, c='#FF9191', marker='.')
    plt.plot(np.linspace(0, 500, 100), np.poly1d(np.polyfit(controlFPRate, controlSens, 4))(np.linspace(0, 500, 100)), 'r--')

    plt.xlabel('Adjusted False Positives per Scan')
    plt.ylabel('Sensitivity')
    plt.legend(labels)
    plt.show()

def getAvgSensN(minFP, maxFP, zippedFPSens):
    # get points that lie between minFP <= zippedFPSens[0] < maxFP
    processedList = [i for i in zippedFPSens if minFP <= i[0] and i[0] < maxFP]
    # return string of format: averageSens (n), segmented_array
    # print(np.array(processedList)[:,1].shape); exit()
    if len(processedList) == 0:
        return 'None', []
    return (str(round(np.mean(processedList, axis=0)[1], 3))+' ('+str(len(processedList))+')'), (np.array(processedList)[:,1])


def bucketize(experimentList, experimentLabels):
    # generate table of sensitivity and augmentation level
    FPBuckets = [[0,100],[100,200],[200,300],[300,400],[400,500],[500,600]]
    table = []
    ttable = []

    # processed all specified experiments
    for experimentPath in experimentList:
        totalFPrate = []
        totalSens = []
        for file in os.listdir(experimentPath)[:10]:
            trialPath = experimentPath + file + '/'
            try:
                totalFPrate.extend(pickle.load(open(trialPath + 'aug_FPratesAdj1.pickle', 'rb')))
                totalSens.extend(pickle.load(open(trialPath + 'aug_sensitivities1.pickle', 'rb')))
            except:
                pass
        tablecol = []
        ttablecol = []
        for i in FPBuckets:
            string, processedList = getAvgSensN(i[0], i[1], zip(totalFPrate, totalSens))
            tablecol.append(string)
            ttablecol.append(processedList)
        table.append(tablecol)
        ttable.append(ttablecol)

    # tests on columns
    # shapiro wilk normality test between each level of augmentation
    shapiro = []
    for i in range(len(ttable)):
        col = []
        for k in range(len(FPBuckets)):
            if len(ttable[i][k]) >= 3:
                col.append(stats.shapiro(ttable[i][k]))
            else:
                col.append(np.nan)
        shapiro.append(col)

    # 2 sample t test performed between each level of augmentation
    tstats = []
    for i in range(len(ttable)-1):
        tstatscol = []
        for k in range(len(FPBuckets)):
            # print(i, len(ttable[i][k]), len(ttable[i + 1][k]))
            if len(ttable[i][k])>=20 and len(ttable[i+1][k])>=20:
                tstatscol.append(stats.mannwhitneyu(ttable[i][k], ttable[i+1][k]))
            else:
                tstatscol.append(np.nan)
        tstats.append(tstatscol)

    # append FP bucket ranges as a column to start of table
    FPRanges = [str(i[0])+'-'+str(i[1]) for i in FPBuckets]
    table.insert(0, FPRanges)
    # extend the labels to account for sensitivities and control
    labels = ['FPs']
    labels.extend(experimentLabels)
    pd.DataFrame(np.array(table).transpose(), columns=labels).to_csv('../desktop/test.csv', index=False)

    # print shapiro wilk test
    print('shapiro wilk')
    print(pd.DataFrame(shapiro).transpose())

    # print t stats
    print('t stats')
    print(pd.DataFrame(tstats).transpose())
    pd.DataFrame(tstats).transpose().to_csv('../desktop/teststats.csv', index=False)

if __name__ == '__main__':
    experimentPath = 'C:/Users/CGAO8/Documents/Deep Learning Practice/images/experiment12/trials/'
    controlsPath = 'C:/Users/CGAO8/Documents/Deep Learning Practice/images/controls/trials/'
    labels = []
    labels.append('Augmented Data')
    labels.append('Control Data')
    labels.append('test scatterplot')
    labels.append('control scatterplot')
    # graphOne(experimentPath, os.listdir(experimentPath))
    # graphManyVsMany(experimentPath, controlsPath, labels)

    bucketize([controlsPath,
               'C:/Users/CGAO8/Documents/Deep Learning Practice/images/experiment13/trials/',
               'C:/Users/CGAO8/Documents/Deep Learning Practice/images/experiment12/trials/',
               'C:/Users/CGAO8/Documents/Deep Learning Practice/images/experiment11/trials/',
               'C:/Users/CGAO8/Documents/Deep Learning Practice/images/experiment14/trials/',
    ], [
        'Control',
        '30% Augmented',
        '50% Augmented',
        '100% Augmented',
        '200% Augmented'
    ])