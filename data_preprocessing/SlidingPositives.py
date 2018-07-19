# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 13:16:59 2017

@author: SMAHESH
"""

import pickle
import gc
import pandas as pd
from collections import OrderedDict
import numpy as np
import random
random.seed(10)

#global / box sizes
Xsize = 40
Ysize = 40
Zsize = 18

XYstride = 20
Zstride = 9

def createInput(listp, dictp):
    listToReturn = []
    minZIndex = listp[2][0]
    maxZIndex = listp[2][1]
    zlist = list(dictp.items())
    for i in range(minZIndex, maxZIndex):
        value = zlist[i][1]
        part = np.array(value)
        slicewanted = part[listp[0][0]:listp[0][1],listp[1][0]:listp[1][1]]
        listToReturn.append(slicewanted)
    return (np.dstack(listToReturn))


x1 = pd.ExcelFile("noduleDimensions.xlsx")
allNodules = x1.parse(x1.sheet_names[0])
allNodules = allNodules.sort_values(['SeriesID'])

IDs = list(allNodules["SeriesID"].drop_duplicates())
validation_set = IDs[-120:-1]
validation_set.append(IDs[-1])

nodulesToUse = x1.parse(x1.sheet_names[2])  
noduleSet = set(nodulesToUse["NoduleID"])
del nodulesToUse
del x1

allboxXs = []
allboxYs = []
Xlow = 0
Xhigh = Xsize
while Xhigh < 512:
    allboxXs.append([Xlow, Xhigh])
    allboxYs.append([Xlow, Xhigh])
    Xlow += XYstride
    Xhigh += XYstride


#headerList = list(allNodules)
prevID = None
positivelist = []
tempdict = None
imageDict = None
takeNegativeSample = True
seriesIDset = set()
print (len(allNodules))
counter = 0
for i in range(len(allNodules)):
    if (allNodules["SliceThickness"][i] <= 2.5):
        nodeID = allNodules["NoduleID"][i]
        seriesID = allNodules["SeriesID"][i] 
            
        if prevID != seriesID and seriesID not in validation_set:
            counter+=1
            if prevID in seriesIDset:
               print ("Repeate Series ID: " + str(prevID)) 
            else:
                seriesIDset.add(prevID)
            if counter % 10 == 0:
                print (str(counter) + " Done")                
            
            filestring = str(seriesID)
            with open(filestring, "rb") as f:
                loadedData = pickle.load(f)
            gc.collect()
            tempdict = loadedData
            imageDict = OrderedDict(sorted(tempdict.items(), key=lambda t: t[0]))
            
            allboxZs = []
            Zlow = 0
            Zhigh = Zsize
            while Zhigh < len(imageDict):
                allboxZs.append([Zlow, Zhigh])
                Zlow += Zstride
                Zhigh += Zstride
            allboxZs.append([len(imageDict) - Zsize, len(imageDict)]) 
        
        prevID = seriesID  
        
        if nodeID in noduleSet:
            #then store in data set
            centerX = allNodules["centerX"][i]
            centerY = allNodules["centerY"][i] 
            centerZ = allNodules["centerZ"][i]
            postoadd = []
            
            if centerZ not in imageDict:
                centerZ *= -1
            if centerZ in imageDict:
                #create all possible boxes
                posZs = []
                posYs = []
                posXs = []
                postoadd = []
                
                for boxZ in allboxZs:
                    zmid = list(imageDict.keys()).index(centerZ) 
                    if boxZ[0] < zmid < boxZ[1]:
                        posZs.append(boxZ)
                for boxY in allboxYs:
                    if boxY[0] < centerY < boxY[1]:
                        posYs.append(boxY)
                for boxX in allboxXs:
                    if boxX[0] < centerX < boxX[1]:
                        posXs.append(boxX)
                for boxZ in posZs:
                    for boxY in posYs:
                        for boxX in posXs:
                            postoadd = createInput([boxX, boxY, boxZ], imageDict)                        
                            positivelist.append(postoadd)
                    

print ("Positive samples: " + str(len(positivelist)))

with open("SlidingBoxPositive.pickle", 'wb') as handle:
    pickle.dump(positivelist, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
