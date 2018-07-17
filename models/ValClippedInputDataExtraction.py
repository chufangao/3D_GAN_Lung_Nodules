# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 12:46:27 2017

@author: SMAHESH
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 17:08:38 2017

@author: SMAHESH
"""

import pickle
import gc
import pandas as pd
from collections import OrderedDict
import numpy as np
import random
random.seed(10)
#Creates and clips sliding-box inputs for validation scans
#Requires:
#    - List of validation series IDs
#    - Dictionaries of image data for each scan
#    - noduleDimensions excel sheet
#    Produces:
#        - All sliding-box subvolumes for validation scans
#        - noduleBoxes dictionary - for each scan, for each nodule, lists coordinates of all
#        sliding-box subvolumes overlapping that nodule
#        - fakeNoduleBoxes dictionary (same thing as noduleBoxes but for low-certainty nodules)
#        - List of number of slices in each scan
#        List of validation scans IDs that are broken (nodule coordinates don’t match dictionary) and
#        working (all coordinates match)
#Creating sliding-box style inputs from validation scans
Xsize = 40
Ysize = 40
Zsize = 18
savePath = '/home/cc/Data/'

XYstride = 20
Zstride = 9

top_threshold = 2446
bottom_threshold = -1434

counterFile = 0
sliceamounts = []

def createInput(listp, dictp):
    # create input 
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

def checkIntersect(nodeID, box, dictp):
    intersect = 0
    nodeData = allNodules.loc[allNodules['NoduleID'] == nodeID]
    xcoords = [nodeData["minimumX"].iloc[0],nodeData["maximumX"].iloc[0]]
    ycoords = [nodeData["minimumY"].iloc[0],nodeData["maximumY"].iloc[0]]
    if nodeData["minimumZ"].iloc[0] in dictp:
        minZIndex = list(dictp.keys()).index(nodeData["minimumZ"].iloc[0])
        maxZIndex = list(dictp.keys()).index(nodeData["maximumZ"].iloc[0])
        zcoords = [minZIndex, maxZIndex]
    else:
        return -1
            
    xmin = box[0][0]
    ymin = box[1][0]
    zmin = box[2][0]

    if xmin in range(xcoords[0] - Xsize, xcoords[1]):
        if ymin in range(ycoords[0] - Ysize, ycoords[1]):
            if zmin >= zcoords[0] - Zsize and zmin <= zcoords[1]:
                intersect = 1                
    return intersect    
   

allboxZs = []
allboxYs = []
allboxXs = []
noduleDict = {}
fakeNoduleDict = {}

seriesIDs = None
with open(savePath+"ValidationIDs4.pickle", "rb") as f:
    seriesIDs = pickle.load(f)
    
x1 = pd.ExcelFile("noduleDimensions.xlsx")
allNodules = x1.parse(x1.sheet_names[0])
validIDs = set(allNodules[allNodules["SliceThickness"] <= 2.5]["SeriesID"])

Xlow = 0
Xhigh = Xsize
while Xhigh < 512:
    allboxXs.append([Xlow, Xhigh])
    allboxYs.append([Xlow, Xhigh])
    Xlow += XYstride
    Xhigh += XYstride
brokenfile = False
brokenlist = []
workinglist = []
# iterate thru seriesIDs
for seriesID in seriesIDs:
    allboxZs = []
    if seriesID in validIDs:    
        filestring = str(seriesID)
        # open pickle file and sort by z value
        with open(savePath+filestring, "rb") as f:
            loadedData = pickle.load(f)
        tempdict = loadedData
        imageDict = OrderedDict(sorted(tempdict.items(), key=lambda t: t[0]))

        # threshold the images
        '''
        for z in imageDict:
            for j in range(len(imageDict[z])):
                for k in range(len(imageDict[z][j])):
                    if imageDict[z][j][k] > top_threshold:
                        imageDict[z][j][k] = top_threshold
                    if imageDict[z][j][k] < bottom_threshold:
                        imageDict[z][j][k] = bottom_threshold
        '''
        testlist = []
        # get all nodules whos series id match validation seriesid
        seriesNodules = allNodules[allNodules["SeriesID"] == seriesID]
        nodules = list(seriesNodules[seriesNodules["certainty"] >= 3]["NoduleID"])
        fakeNodules = list(seriesNodules[seriesNodules["certainty"] < 3]["NoduleID"])
        noduleDict[seriesID] = {}
        fakeNoduleDict[seriesID] = {}
        for node in nodules:
            noduleDict[seriesID][node] = []
        for node in fakeNodules:
            fakeNoduleDict[seriesID][node] = []
        
        Zlow = 0
        Zhigh = Zsize
        while Zhigh < len(imageDict):
            allboxZs.append([Zlow, Zhigh])
            Zlow += Zstride
            Zhigh += Zstride
        allboxZs.append([len(imageDict) - Zsize, len(imageDict)])  
        
        for boxZ in allboxZs:
            for boxY in allboxYs:
                for boxX in allboxXs:
                    box = [boxX, boxY, boxZ]
                    testlist.append(createInput(box, imageDict))
                    
                    for nodeID in nodules:
                        checkedint = checkIntersect(nodeID, box, imageDict)
                        if checkedint == 1:
                            noduleDict[seriesID][nodeID].append(box)
                        elif checkedint == -1:
                            brokenfile = True
                            break
                    
                    for nodeID in fakeNodules:
                        checkedint = checkIntersect(nodeID, box, imageDict)
                        if checkedint == 1:
                            fakeNoduleDict[seriesID][nodeID].append(box)
                        elif checkedint == -1:
                            brokenfile = True
                            break
                            
                    if brokenfile:
                        break
                if brokenfile:
                    break
            if brokenfile:
                break
    
        if (not brokenfile):
            sliceamounts.append(len(imageDict))
            workinglist.append(seriesID)                  
            with open(savePath+"ValClipped" + filestring + ".pickle", 'wb') as handle:
                pickle.dump(testlist, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            brokenlist.append(seriesID)
        counterFile += 1
        print ("File: " + str(counterFile))    
        brokenfile = False            

print ("Broken Files: " + str(len(brokenlist)))
print ("Working Files: " + str(len(workinglist)))              
with open(savePath+"noduleBoxes.pickle", 'wb') as handle:
    pickle.dump(noduleDict, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(savePath+"fakeNoduleBoxes.pickle", 'wb') as handle:
    pickle.dump(fakeNoduleDict, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(savePath+"brokenValidationSeries.pickle", 'wb') as handle:
    pickle.dump(brokenlist, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(savePath+"workingValidationSeries.pickle", 'wb') as handle:
    pickle.dump(workinglist, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(savePath+"sliceamount.pickle", 'wb') as handle:
    pickle.dump(sliceamounts, handle, protocol=pickle.HIGHEST_PROTOCOL)
